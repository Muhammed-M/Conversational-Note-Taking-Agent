"""
agent.py — The brain of the note-taking assistant.

This module handles the full conversation flow. Every time the user sends a message,
agent.run() is called. It looks at the current state (mode) and decides what to do.

The 4 main flows:
  1. Save    — user wants to create a new note
  2. Search  — user wants to find notes (keyword / tag / semantic)
  3. Update  — user wants to change an existing note (needs human confirmation)
  4. Delete  — user wants to remove a note (needs human confirmation)

For update and delete, we use a "human-in-the-loop" pattern:
  - If multiple notes match → ask user which one (AWAITING_DISAMBIGUATION)
  - Once note is selected  → show what will happen and ask yes/no (AWAITING_CONFIRM)
"""

import config
import llm
from models import Note
from schemas import IntentResult, RewrittenNote
from state import AgentState
from store import NoteStore
from vector_store import VectorStore


class Agent:
    """The conversational note-taking agent."""

    def __init__(self, store: NoteStore, vector_store: VectorStore):
        self.store = store
        self.vector_store = vector_store

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point — called once per user message
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, user_message: str, state: AgentState) -> AgentState:
        """
        Main entry point. Called every time the user sends a message.

        Adds the message to memory, then routes based on current mode:
          - IDLE                  → parse intent and act
          - AWAITING_DISAMBIGUATION → user is picking which note they meant
          - AWAITING_CONFIRM      → user is saying yes/no to confirm an action
        """
        # Save this message to conversation memory
        state["messages"].append({"role": "user", "content": user_message})

        mode = state.get("mode", "IDLE")

        if mode == "AWAITING_DISAMBIGUATION":
            return self._resolve_disambiguation(user_message, state)

        if mode == "AWAITING_CONFIRM":
            return self._resolve_confirmation(user_message, state)

        # Normal mode — figure out what the user wants
        return self._handle_new_message(user_message, state)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Parse intent and route to the right flow
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_new_message(self, user_message: str, state: AgentState) -> AgentState:
        """
        Ask the LLM what the user wants, then call the correct handler.

        llm.pick_intent() returns a typed IntentResult object (defined in schemas.py).
        We read result.intent to decide which handler to call,
        and pass the whole result object so each handler can access its fields directly.
        """
        result = llm.pick_intent(user_message, state["messages"])

        if result.intent == "save":
            return self._save_note(result, state)

        elif result.intent == "search_keyword":
            return self._search_by_keyword(result, state)

        elif result.intent == "search_tags":
            return self._search_by_tags(result, state)

        elif result.intent == "search_semantic":
            return self._search_semantic(result, state)

        elif result.intent == "update":
            return self._start_update(result, user_message, state)

        elif result.intent == "delete":
            return self._start_delete(result, state)

        else:
            state["final_response"] = (
                "I'm your note-taking assistant. I can save, search, update, or delete your notes. What would you like to do?"
            )
            return state

    # ─────────────────────────────────────────────────────────────────────────
    # Flow 1 — Save a new note
    # ─────────────────────────────────────────────────────────────────────────

    def _save_note(self, result: IntentResult, state: AgentState) -> AgentState:
        """
        Save a new note to both stores.

        The LLM already extracted title, body, and tags in pick_intent().
        They are available directly as typed attributes on the IntentResult object.

        Steps:
          1. Save to SQLite (gets an auto-generated ID)
          2. Embed and save to Qdrant (for future semantic search)
        """
        title = (result.title or "").strip()
        body = (result.body or "").strip()
        tags = result.tags or []

        if not title or not body:
            state["final_response"] = "Please give me more detail about the note you'd like to save."
            return state

        # Save to SQLite
        note = self.store.add_note(title=title, body=body, tags=tags)

        # Save embedding to Qdrant so it's searchable semantically later
        self.vector_store.upsert_note(note)

        # Remember this note's ID for multi-turn references like "update that note"
        state["last_note_id"] = note.id

        tags_text = f" | Tags: {', '.join(note.tags)}" if note.tags else ""
        state["final_response"] = f"✅ Saved: '{note.title}' (ID: {note.short_id}){tags_text}"
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Flow 2 — Search notes (3 tools, LLM picks which one)
    # ─────────────────────────────────────────────────────────────────────────

    def _search_by_keyword(self, result: IntentResult, state: AgentState) -> AgentState:
        """
        Search SQLite for notes containing a specific keyword.
        Used when the user mentions a specific word to find.
        Example: "find notes with the word Python"
        """
        keyword = (result.keyword or "").strip()
        if not keyword:
            state["final_response"] = "Please tell me the keyword you'd like to search for."
            return state

        notes = self.store.search_by_keyword(keyword, top_n=config.TOP_K_KEYWORD)
        return self._format_search_results(notes, f"keyword '{keyword}'", state)

    def _search_by_tags(self, result: IntentResult, state: AgentState) -> AgentState:
        """
        Search SQLite for notes that have specific tags.
        Used when the user filters by category.
        Example: "show me notes tagged work"
        """
        tags = result.tags or []
        if not tags:
            state["final_response"] = "Please tell me which tags to search by."
            return state

        notes = self.store.search_by_tags(tags, top_n=config.TOP_K_TAG)
        return self._format_search_results(notes, f"tags: {', '.join(tags)}", state)

    def _search_semantic(self, result: IntentResult, state: AgentState) -> AgentState:
        """
        Semantic search using Qdrant vector similarity.
        Used for natural language questions.
        Example: "what did I write about the team meeting last week?"

        Steps:
          1. Embed the query → get a vector
          2. Search Qdrant for similar vectors → get a list of note IDs
          3. Fetch full note data from SQLite using those IDs
        """
        query = (result.query or "").strip()
        if not query:
            state["final_response"] = "Please describe what you're looking for."
            return state

        # Search Qdrant — returns a list of note IDs ranked by similarity
        note_ids = self.vector_store.search(query, top_k=config.TOP_K_SEMANTIC)

        # Fetch the full notes from SQLite using those IDs
        notes = [self.store.get_note_by_id(nid) for nid in note_ids]
        notes = [n for n in notes if n is not None]  # filter out any IDs not found in SQLite

        return self._format_search_results(notes, "your query", state)

    def _format_search_results(self, notes: list[Note], search_desc: str, state: AgentState) -> AgentState:
        """Format a list of notes into a readable response string."""
        if not notes:
            state["final_response"] = f"No notes found for {search_desc}."
            return state

        # Remember the first result so the user can reference it later (e.g. "update that note")
        state["last_note_id"] = notes[0].id

        lines = []
        for note in notes:
            tags_text = f" | Tags: {', '.join(note.tags)}" if note.tags else ""
            lines.append(f"• [{note.short_id}] {note.title}{tags_text}\n  {note.body}")

        state["final_response"] = f"Found {len(notes)} note(s):\n\n" + "\n\n".join(lines)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Flow 3 — Update a note (with human-in-the-loop)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_update(self, result: IntentResult, user_message: str, state: AgentState) -> AgentState:
        """
        Start the update flow: find matching notes and ask the user to pick one.
        We save the original user_message because it contains the full update instruction
        (e.g. "change the standup to Wednesdays") — we'll need it later when
        we call the LLM to actually rewrite the note.
        """
        query = result.query or user_message   # result.query has what to search for
        candidates = self._find_candidates(query)

        if not candidates:
            state["final_response"] = "No notes found matching your request. Nothing to update."
            return state

        # Store the intent and the original instruction for later use
        state["pending_action"] = {
            "intent": "update",
            "user_instruction": user_message,  # saved so we can call llm.rewrite_note() later
        }

        return self._ask_user_to_pick(candidates, "update", state)

    # ─────────────────────────────────────────────────────────────────────────
    # Flow 4 — Delete a note (with human-in-the-loop)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_delete(self, result: IntentResult, state: AgentState) -> AgentState:
        """Start the delete flow: find matching notes and ask the user to pick one."""
        query = result.query or ""   # result.query describes which note to delete
        candidates = self._find_candidates(query)

        if not candidates:
            state["final_response"] = "No notes found matching your request. Nothing to delete."
            return state

        state["pending_action"] = {"intent": "delete"}

        return self._ask_user_to_pick(candidates, "delete", state)

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helper — find candidate notes across both stores
    # ─────────────────────────────────────────────────────────────────────────

    def _find_candidates(self, query: str) -> list[Note]:
        """
        Search both stores to find notes that could match the user's request.
        We use both vector search (semantic) and keyword search (SQL) and merge results.
        Returns at most TOP_K_CANDIDATES unique notes.
        """
        # Semantic search in Qdrant (finds notes similar in meaning)
        vector_ids = self.vector_store.search(query, top_k=config.TOP_K_CANDIDATES)

        # Keyword search in SQLite (finds notes containing the exact words)
        keyword_notes = self.store.search_by_keyword(query, top_n=config.TOP_K_CANDIDATES)

        # Merge both result sets, removing duplicates, keeping order
        seen_ids = set()
        candidates = []

        for note_id in vector_ids:
            note = self.store.get_note_by_id(note_id)
            if note and note.id not in seen_ids:
                candidates.append(note)
                seen_ids.add(note.id)

        for note in keyword_notes:
            if note.id not in seen_ids:
                candidates.append(note)
                seen_ids.add(note.id)

        return candidates[:config.TOP_K_CANDIDATES]

    def _ask_user_to_pick(self, candidates: list[Note], intent: str, state: AgentState) -> AgentState:
        """
        Present matching candidates to the user.

        If only 1 note found:
          - For delete: ask yes/no directly (AWAITING_CONFIRM)
          - For update: call the LLM to preview the rewritten note, then ask yes/no

        If multiple notes found:
          - Show a numbered list and ask user to pick (AWAITING_DISAMBIGUATION)
        """
        # Save candidates in state so we can reference them during disambiguation
        state["search_candidates"] = [n.to_dict() for n in candidates]

        if len(candidates) == 1:
            note = candidates[0]
            state["pending_action"]["note_id"] = note.id
            state["pending_action"]["note_title"] = note.title

            if intent == "delete":
                state["final_response"] = (
                    f"Found: '{note.title}' (ID: {note.short_id})\n\n"
                    f"Are you sure you want to delete this note? (yes / no)"
                )

            else:  # update
                # Call the LLM to preview what the updated note will look like
                user_instruction = state["pending_action"].get("user_instruction", "")
                rewritten = llm.rewrite_note(note, user_instruction)  # returns RewrittenNote
                # Store as dict so it can live in the state (AgentState uses plain dicts)
                state["pending_action"]["updated_fields"] = rewritten.model_dump()

                state["final_response"] = (
                    f"Here is how '{note.title}' will look after the update:\n\n"
                    f"  Title : {rewritten.title}\n"
                    f"  Body  : {rewritten.body}\n"
                    f"  Tags  : {', '.join(rewritten.tags)}\n\n"
                    f"Confirm update? (yes / no)"
                )

            state["mode"] = "AWAITING_CONFIRM"
            return state

        # Multiple matches — ask user to choose
        options = "\n".join(
            [f"  [{i + 1}] '{n.title}' (ID: {n.short_id})" for i, n in enumerate(candidates)]
        )
        state["final_response"] = (
            f"Found {len(candidates)} notes that could match. Which one did you mean?\n\n"
            f"{options}\n\n"
            f"Reply with the number (e.g. 1) or 'cancel'."
        )
        state["mode"] = "AWAITING_DISAMBIGUATION"
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Mode: AWAITING_DISAMBIGUATION — user picks which note they meant
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_disambiguation(self, user_message: str, state: AgentState) -> AgentState:
        """
        Handle the user's reply when we asked them to pick from a list of notes.
        Accepts a number ("1", "2", "3") or a title snippet.
        After picking, moves to AWAITING_CONFIRM.
        """
        msg = user_message.strip().lower()

        if msg in ("cancel", "no", "stop", "never mind"):
            return self._cancel(state)

        candidates = state.get("search_candidates", [])
        pending = state.get("pending_action", {})
        intent = pending.get("intent", "update")

        picked = None

        # Match by number: user typed "1", "2", or "3"
        if msg.isdigit():
            idx = int(msg) - 1  # convert to 0-based index
            if 0 <= idx < len(candidates):
                picked = candidates[idx]

        # Match by title snippet: user typed part of the note title
        if picked is None:
            for candidate in candidates:
                title_lower = candidate["title"].lower()
                if title_lower in msg or msg in title_lower:
                    picked = candidate
                    break

        if picked is None:
            state["final_response"] = (
                f"Didn't understand. Please reply with a number between 1 and {len(candidates)}, "
                f"or type 'cancel'."
            )
            return state

        # User has selected a note — save it to pending_action
        state["pending_action"]["note_id"] = picked["id"]
        state["pending_action"]["note_title"] = picked["title"]

        note = self.store.get_note_by_id(picked["id"])

        if intent == "delete":
            state["final_response"] = (
                f"Are you sure you want to delete '{picked['title']}' (ID: {picked['id'][:8]})? (yes / no)"
            )

        else:  # update
            # Now that we know which note the user picked, ask the LLM to preview the update
            user_instruction = pending.get("user_instruction", "")
            rewritten = llm.rewrite_note(note, user_instruction)  # returns RewrittenNote
            # Store as dict so it can live in the state (AgentState uses plain dicts)
            state["pending_action"]["updated_fields"] = rewritten.model_dump()

            state["final_response"] = (
                f"Here is how '{note.title}' will look after the update:\n\n"
                f"  Title : {rewritten.title}\n"
                f"  Body  : {rewritten.body}\n"
                f"  Tags  : {', '.join(rewritten.tags)}\n\n"
                f"Confirm update? (yes / no)"
            )

        state["mode"] = "AWAITING_CONFIRM"
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Mode: AWAITING_CONFIRM — user says yes or no
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_confirmation(self, user_message: str, state: AgentState) -> AgentState:
        """
        Handle the user's yes/no response to a pending update or delete.
        If yes → execute the action on both stores.
        If no  → cancel and go back to normal.
        """
        msg = user_message.strip().lower()

        if msg in ("cancel", "no", "n", "nope", "never mind"):
            return self._cancel(state)

        if msg not in ("yes", "y", "sure", "ok", "confirm"):
            state["final_response"] = "Please reply with 'yes' to confirm or 'no' to cancel."
            return state

        # User confirmed — get the details from pending_action
        pending = state.get("pending_action", {})
        intent = pending.get("intent")
        note_id = pending.get("note_id")

        if intent == "delete":
            # Delete from SQLite
            deleted = self.store.delete_note(note_id)
            # Delete embedding from Qdrant
            self.vector_store.delete_note(note_id)

            state["mode"] = "IDLE"
            state["pending_action"] = None
            state["search_candidates"] = None

            if deleted:
                state["final_response"] = f"🗑️ Deleted '{pending.get('note_title', '')}' successfully."
            else:
                state["final_response"] = "Could not find that note. It may have already been deleted."

        elif intent == "update":
            updated_fields = pending.get("updated_fields", {})
            note = self.store.get_note_by_id(note_id)

            if not note:
                state["final_response"] = "Could not find that note to update."
                state["mode"] = "IDLE"
                state["pending_action"] = None
                return state

            # Update in SQLite with the fields the LLM rewrote
            updated_note = self.store.update_note(
                note_id=note_id,
                title=updated_fields.get("title", note.title),
                body=updated_fields.get("body", note.body),
                tags=updated_fields.get("tags", note.tags),
            )

            # Re-embed and update in Qdrant so future semantic searches reflect the new content
            self.vector_store.upsert_note(updated_note)

            state["mode"] = "IDLE"
            state["pending_action"] = None
            state["search_candidates"] = None
            state["last_note_id"] = updated_note.id
            state["final_response"] = f"✅ Updated '{updated_note.title}' (ID: {updated_note.short_id})."

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Helper — cancel any pending action
    # ─────────────────────────────────────────────────────────────────────────

    def _cancel(self, state: AgentState) -> AgentState:
        """Reset to IDLE and tell the user the action was cancelled."""
        state["mode"] = "IDLE"
        state["pending_action"] = None
        state["search_candidates"] = None
        state["final_response"] = "Action cancelled. What else can I help you with?"
        return state
