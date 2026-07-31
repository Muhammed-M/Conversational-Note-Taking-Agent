"""
agent.py — The brain of the note-taking assistant.
"""

from src import config
from src import llm
from src.models import Note
from src.schemas import IntentResult, RewrittenNote
from src.state import AgentState
from src.store import NoteStore
from src.vector_store import VectorStore


class Agent:
    """The conversational note-taking agent."""

    def __init__(self, store: NoteStore, vector_store: VectorStore):
        self.store = store
        self.vector_store = vector_store

    def run(self, user_message: str, state: AgentState) -> AgentState:
        """Main entry point. Called every time user sends a message."""
        state["messages"].append({"role": "user", "content": user_message})

        mode = state.get("mode", "IDLE")

        if mode == "AWAITING_DISAMBIGUATION":
            return self._resolve_disambiguation(user_message, state)

        if mode == "AWAITING_CONFIRM":
            return self._resolve_confirmation(user_message, state)

        return self._handle_new_message(user_message, state)

    def _handle_new_message(self, user_message: str, state: AgentState) -> AgentState:
        """Parse intent via LLM and call correct handler."""
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

        elif result.intent == "chitchat":
            state["final_response"] = (
                result.chitchat_response
                or "Hello! How can I help you manage your notes today?"
            )
            return state

        else:
            state["final_response"] = (
                "I'm your note-taking assistant. I can save, search, update, or delete your notes. What would you like to do?"
            )
            return state

    # ── Flow 1 — Save ─────────────────────────────────────────────────────────

    def _save_note(self, result: IntentResult, state: AgentState) -> AgentState:
        """Save a new note to both stores."""
        title = (result.title or "").strip()
        body = (result.body or "").strip()
        tags = result.tags or []

        if not title or not body:
            state["final_response"] = "Please give me more detail about the note you'd like to save."
            return state

        note = self.store.add_note(title=title, body=body, tags=tags)
        self.vector_store.upsert_note(note)

        state["last_note_id"] = note.id
        tags_text = f" | Tags: {', '.join(note.tags)}" if note.tags else ""
        state["final_response"] = f"✅ Saved: '{note.title}' (ID: {note.short_id}){tags_text}"
        return state

    # ── Flow 2 — Search ───────────────────────────────────────────────────────

    def _search_by_keyword(self, result: IntentResult, state: AgentState) -> AgentState:
        """Search SQLite by keyword."""
        keyword = (result.keyword or "").strip()
        if not keyword:
            state["final_response"] = "Please tell me the keyword you'd like to search for."
            return state

        notes = self.store.search_by_keyword(keyword, top_n=config.TOP_K_KEYWORD)
        return self._format_search_results(notes, f"keyword '{keyword}'", state)

    def _search_by_tags(self, result: IntentResult, state: AgentState) -> AgentState:
        """Search SQLite by tags."""
        tags = result.tags or []
        if not tags:
            state["final_response"] = "Please tell me which tags to search by."
            return state

        notes = self.store.search_by_tags(tags, top_n=config.TOP_K_TAG)
        return self._format_search_results(notes, f"tags: {', '.join(tags)}", state)

    def _search_semantic(self, result: IntentResult, state: AgentState) -> AgentState:
        """Semantic vector search in Qdrant."""
        query = (result.query or "").strip()
        if not query:
            state["final_response"] = "Please describe what you're looking for."
            return state

        note_ids = self.vector_store.search(query, top_k=config.TOP_K_SEMANTIC)
        notes = [self.store.get_note_by_id(nid) for nid in note_ids]
        notes = [n for n in notes if n is not None]

        return self._format_search_results(notes, "your query", state)

    def _format_search_results(self, notes: list[Note], search_desc: str, state: AgentState) -> AgentState:
        """Format matching notes into response string."""
        if not notes:
            state["final_response"] = f"No notes found for {search_desc}."
            return state

        state["last_note_id"] = notes[0].id
        lines = []
        for note in notes:
            tags_text = f" | Tags: {', '.join(note.tags)}" if note.tags else ""
            lines.append(f"• [{note.short_id}] {note.title}{tags_text}\n  {note.body}")

        state["final_response"] = f"Found {len(notes)} note(s):\n\n" + "\n\n".join(lines)
        return state

    # ── Flow 3 — Update ───────────────────────────────────────────────────────

    def _start_update(self, result: IntentResult, user_message: str, state: AgentState) -> AgentState:
        """Find matching notes and start update disambiguation/confirmation."""
        query = result.query or user_message
        candidates = self._find_candidates(query)

        if not candidates:
            state["final_response"] = "No notes found matching your request. Nothing to update."
            return state

        state["pending_action"] = {
            "intent": "update",
            "user_instruction": user_message,
        }

        return self._ask_user_to_pick(candidates, "update", state)

    # ── Flow 4 — Delete ───────────────────────────────────────────────────────

    def _start_delete(self, result: IntentResult, state: AgentState) -> AgentState:
        """Find matching notes and start delete disambiguation/confirmation."""
        query = result.query or ""
        candidates = self._find_candidates(query)

        if not candidates:
            state["final_response"] = "No notes found matching your request. Nothing to delete."
            return state

        state["pending_action"] = {"intent": "delete"}
        return self._ask_user_to_pick(candidates, "delete", state)

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _find_candidates(self, query: str) -> list[Note]:
        """Find candidate notes across vector search + keyword search."""
        vector_ids = self.vector_store.search(query, top_k=config.TOP_K_CANDIDATES)
        keyword_notes = self.store.search_by_keyword(query, top_n=config.TOP_K_CANDIDATES)

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
        """Present candidates to user for selection or confirm."""
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
            else:
                user_instruction = state["pending_action"].get("user_instruction", "")
                rewritten = llm.rewrite_note(note, user_instruction)
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

    def _resolve_disambiguation(self, user_message: str, state: AgentState) -> AgentState:
        """Handle user choice from candidate list."""
        msg = user_message.strip().lower()

        if msg in ("cancel", "no", "stop", "never mind"):
            return self._cancel(state)

        candidates = state.get("search_candidates", [])
        pending = state.get("pending_action", {})
        intent = pending.get("intent", "update")
        picked = None

        if msg.isdigit():
            idx = int(msg) - 1
            if 0 <= idx < len(candidates):
                picked = candidates[idx]

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

        state["pending_action"]["note_id"] = picked["id"]
        state["pending_action"]["note_title"] = picked["title"]
        note = self.store.get_note_by_id(picked["id"])

        if intent == "delete":
            state["final_response"] = (
                f"Are you sure you want to delete '{picked['title']}' (ID: {picked['id'][:8]})? (yes / no)"
            )
        else:
            user_instruction = pending.get("user_instruction", "")
            rewritten = llm.rewrite_note(note, user_instruction)
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

    def _resolve_confirmation(self, user_message: str, state: AgentState) -> AgentState:
        """Handle user confirmation (yes/no) for pending update/delete."""
        msg = user_message.strip().lower()

        if msg in ("cancel", "no", "n", "nope", "never mind"):
            return self._cancel(state)

        if msg not in ("yes", "y", "sure", "ok", "confirm"):
            state["final_response"] = "Please reply with 'yes' to confirm or 'no' to cancel."
            return state

        pending = state.get("pending_action", {})
        intent = pending.get("intent")
        note_id = pending.get("note_id")

        if intent == "delete":
            deleted = self.store.delete_note(note_id)
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

            updated_note = self.store.update_note(
                note_id=note_id,
                title=updated_fields.get("title", note.title),
                body=updated_fields.get("body", note.body),
                tags=updated_fields.get("tags", note.tags),
            )

            self.vector_store.upsert_note(updated_note)

            state["mode"] = "IDLE"
            state["pending_action"] = None
            state["search_candidates"] = None
            state["last_note_id"] = updated_note.id
            state["final_response"] = f"✅ Updated '{updated_note.title}' (ID: {updated_note.short_id})."

        return state

    def _cancel(self, state: AgentState) -> AgentState:
        """Reset state and inform user."""
        state["mode"] = "IDLE"
        state["pending_action"] = None
        state["search_candidates"] = None
        state["final_response"] = "Action cancelled. What else can I help you with?"
        return state
