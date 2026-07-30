"""
LangGraph State Machine for Conversational Note-Taking Agent.

Manages conversational state transitions, intent parsing, disambiguation of
multiple matching candidates, explicit user confirmation for destructive actions,
and multi-turn context (last_note_id anaphora resolution).
"""

import os
import re
from typing import Any, Optional
from models import Note
from state import AgentState
from store import NoteStore
from tools import AddNoteInput, DeleteNoteInput, SearchNotesInput, UpdateNoteInput
from vector_store import VectorNoteStore


class NoteAgentGraph:
    """
    State machine executor handling conversation flow and safety gates.
    """

    def __init__(self, store: NoteStore, vector_store: VectorNoteStore):
        self.store = store
        self.vector_store = vector_store
        
        # Wire automatic vector store synchronization on SQLite mutations
        self.store.register_on_created(self.vector_store.upsert_note)
        self.store.register_on_updated(self.vector_store.upsert_note)
        self.store.register_on_deleted(self.vector_store.delete_note)

        self.api_key = os.environ.get("GEMINI_API_KEY")

    def run(self, user_message: str, state: AgentState) -> AgentState:
        """
        Main entry point to step the state machine with a new user message.
        """
        # Append user message to conversation history
        state["messages"].append({"role": "user", "content": user_message})

        mode = state.get("mode", "IDLE")

        # 1. Routing based on current mode
        if mode in ("AWAITING_CONFIRM", "AWAITING_DISAMBIGUATION"):
            return self._resolve_pending(user_message, state)

        # 2. Operational state is IDLE -> parse intent and parameters
        return self._agent_step(user_message, state)

    def _agent_step(self, user_message: str, state: AgentState) -> AgentState:
        """
        Parse user intent via Gemini LLM or fallback deterministic intent parser,
        handling 'that note' / 'last note' multi-turn references.
        """
        parsed = self._parse_intent(user_message, state)
        intent = parsed.get("intent")
        args = parsed.get("args", {})

        if intent == "add":
            # Add operations do not require confirmation gate; execute directly
            title = args.get("title", "")
            body = args.get("body", "")
            tags = args.get("tags", [])

            if not title or not body:
                state["final_response"] = "Please provide both a title and details for the note you'd like to save."
                return state

            new_note = self.store.add_note(title=title, body=body, tags=tags)
            state["last_note_id"] = new_note.id
            state["final_response"] = (
                f"Saved note '{new_note.title}' (ID: {new_note.short_id})"
                + (f" with tags: {', '.join(new_note.tags)}" if new_note.tags else ".")
            )
            return state

        elif intent == "search":
            query = args.get("query")
            tags = args.get("tags")
            return self._execute_search_and_respond(query, tags, state, target_action="search")

        elif intent == "update":
            note_id = args.get("note_id")
            query = args.get("query") or user_message
            tags = args.get("tags")

            # Check for multi-turn reference ("that note", "the last note")
            if not note_id and state.get("last_note_id"):
                if any(w in user_message.lower() for w in ["that note", "last note", "this note", "it"]):
                    note_id = state["last_note_id"]

            updates = {
                "title": args.get("title"),
                "body": args.get("body"),
                "tags": args.get("tags"),
            }

            return self._prepare_mutation("update", note_id, query, tags, updates, state)

        elif intent == "delete":
            note_id = args.get("note_id")
            query = args.get("query") or user_message

            # Check multi-turn reference
            if not note_id and state.get("last_note_id"):
                if any(w in user_message.lower() for w in ["that note", "last note", "this note", "it"]):
                    note_id = state["last_note_id"]

            return self._prepare_mutation("delete", note_id, query, None, {}, state)

        else:
            # Conversational reply fallback
            state["final_response"] = (
                "I am your note-taking assistant. You can ask me to save, search, update, or delete notes."
            )
            return state

    def _prepare_mutation(
        self,
        intent: str,
        note_id: Optional[str],
        query: Optional[str],
        tags: Optional[list[str]],
        updates: dict[str, Any],
        state: AgentState,
    ) -> AgentState:
        """
        Prepares an update or delete action by finding matching candidates
        and applying Disambiguation or Confirmation gates.
        """
        target_note: Optional[Note] = None

        if note_id:
            target_note = self.store.get_note_by_id(note_id)

        if target_note:
            candidates = [target_note]
        else:
            # Perform search to find candidate notes
            vector_ids = self.vector_store.search(query or "", top_k=5)
            sql_notes = self.store.search_notes_sql(query=query, tags=tags)

            # Combine and deduplicate candidates
            candidate_map: dict[str, Note] = {}
            for vid in vector_ids:
                n = self.store.get_note_by_id(vid)
                if n:
                    candidate_map[n.id] = n
            for n in sql_notes:
                candidate_map[n.id] = n

            candidates = list(candidate_map.values())

        if len(candidates) == 0:
            state["final_response"] = f"No notes found matching '{query}' to {intent}."
            state["mode"] = "IDLE"
            return state

        if len(candidates) > 1:
            # INTENT DISAMBIGUATION GATE: Multiple notes match user request
            state["mode"] = "AWAITING_DISAMBIGUATION"
            state["search_candidates"] = [n.to_dict() for n in candidates]
            state["pending_action"] = {
                "intent": intent,
                "updates": updates,
            }

            options_text = "\n".join(
                [f"[{idx+1}] '{n.title}' (ID: {n.short_id}): {n.body[:60]}..." for idx, n in enumerate(candidates)]
            )
            state["final_response"] = (
                f"Multiple notes matched your request to {intent}. Which one did you mean?\n{options_text}\n"
                "Reply with the number (e.g. 1) or 'cancel'."
            )
            return state

        # Single candidate identified -> CONFIRMATION GATE
        selected_note = candidates[0]
        state["mode"] = "AWAITING_CONFIRM"
        state["pending_action"] = {
            "intent": intent,
            "note_id": selected_note.id,
            "note_title": selected_note.title,
            "updates": updates,
        }

        if intent == "delete":
            action_desc = f"DELETE note '{selected_note.title}' (ID: {selected_note.short_id})"
        else:
            action_desc = f"UPDATE note '{selected_note.title}' (ID: {selected_note.short_id})"

        state["final_response"] = f"Are you sure you want to {action_desc}? (yes / no)"
        return state

    def _resolve_pending(self, user_message: str, state: AgentState) -> AgentState:
        """
        Handle user responses when agent is paused at a Confirmation or Disambiguation gate.
        """
        msg_clean = user_message.strip().lower()
        mode = state.get("mode")

        if msg_clean in ("cancel", "no", "n", "never mind", "stop"):
            state["mode"] = "IDLE"
            state["pending_action"] = None
            state["search_candidates"] = None
            state["final_response"] = "Action cancelled."
            return state

        if mode == "AWAITING_DISAMBIGUATION":
            candidates = state.get("search_candidates", [])
            pending = state.get("pending_action", {})

            # Try parsing integer pick (e.g., "1" or "option 2")
            picked_idx = None
            digits = re.findall(r"\d+", msg_clean)
            if digits:
                idx = int(digits[0]) - 1
                if 0 <= idx < len(candidates):
                    picked_idx = idx

            if picked_idx is None:
                state["final_response"] = (
                    f"Invalid selection. Please reply with a number between 1 and {len(candidates)}, or 'cancel'."
                )
                return state

            selected_dict = candidates[picked_idx]
            selected_id = selected_dict["id"]
            selected_title = selected_dict["title"]
            intent = pending.get("intent", "update")

            # Transition from Disambiguation -> Confirmation Gate
            state["mode"] = "AWAITING_CONFIRM"
            state["pending_action"] = {
                "intent": intent,
                "note_id": selected_id,
                "note_title": selected_title,
                "updates": pending.get("updates", {}),
            }

            action_desc = f"{intent.upper()} note '{selected_title}' (ID: {selected_id[:8]})"
            state["final_response"] = f"Are you sure you want to {action_desc}? (yes / no)"
            return state

        elif mode == "AWAITING_CONFIRM":
            if msg_clean in ("yes", "y", "sure", "confirm", "ok"):
                pending = state.get("pending_action", {})
                intent = pending.get("intent")
                note_id = pending.get("note_id")
                updates = pending.get("updates", {})

                if intent == "delete":
                    success = self.store.delete_note(note_id)
                    state["mode"] = "IDLE"
                    state["pending_action"] = None
                    if success:
                        if state.get("last_note_id") == note_id:
                            state["last_note_id"] = None
                        state["final_response"] = "Note deleted successfully."
                    else:
                        state["final_response"] = "Failed to delete note. It may have already been removed."
                    return state

                elif intent == "update":
                    updated = self.store.update_note(
                        note_id=note_id,
                        title=updates.get("title"),
                        body=updates.get("body"),
                        tags=updates.get("tags"),
                    )
                    state["mode"] = "IDLE"
                    state["pending_action"] = None
                    if updated:
                        state["last_note_id"] = updated.id
                        state["final_response"] = f"Updated note '{updated.title}' (ID: {updated.short_id})."
                    else:
                        state["final_response"] = "Failed to update note."
                    return state

            else:
                state["final_response"] = "Please confirm with 'yes' or 'no', or type 'cancel'."
                return state

        state["mode"] = "IDLE"
        return state

    def _execute_search_and_respond(
        self, query: Optional[str], tags: Optional[list[str]], state: AgentState, target_action: str
    ) -> AgentState:
        """Execute hybrid search across SQLite and Qdrant vector index."""
        vector_ids = self.vector_store.search(query or "", top_k=5) if query else []
        sql_notes = self.store.search_notes_sql(query=query, tags=tags)

        # Merge results
        result_map: dict[str, Note] = {}
        for vid in vector_ids:
            n = self.store.get_note_by_id(vid)
            if n:
                result_map[n.id] = n
        for n in sql_notes:
            result_map[n.id] = n

        notes = list(result_map.values())

        if not notes:
            state["final_response"] = f"No notes found matching your search."
            return state

        # Record last viewed note for multi-turn context
        state["last_note_id"] = notes[0].id

        formatted = []
        for n in notes:
            tags_str = f" [tags: {', '.join(n.tags)}]" if n.tags else ""
            formatted.append(f"• '{n.title}' (ID: {n.short_id}){tags_str}\n  {n.body}")

        state["final_response"] = f"Found {len(notes)} note(s):\n" + "\n\n".join(formatted)
        return state

    def _parse_intent(self, text: str, state: AgentState) -> dict[str, Any]:
        """
        Parse natural language input into structured intent dict.
        Uses Gemini LLM tool-calling if GEMINI_API_KEY is active,
        otherwise uses deterministic regex parsing rules.
        """
        if self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                )

                tools = [AddNoteInput, SearchNotesInput, UpdateNoteInput, DeleteNoteInput]
                llm_with_tools = llm.bind_tools(tools)
                res = llm_with_tools.invoke(text)
                if res.tool_calls:
                    tc = res.tool_calls[0]
                    name = tc["name"].lower()
                    args = tc["args"]
                    if "add" in name:
                        return {"intent": "add", "args": args}
                    elif "search" in name or "list" in name:
                        return {"intent": "search", "args": args}
                    elif "update" in name or "modify" in name:
                        return {"intent": "update", "args": args}
                    elif "delete" in name or "remove" in name:
                        return {"intent": "delete", "args": args}
            except Exception as e:
                print(f"[Warning] Gemini API tool parsing fallback triggered: {e}")

        # Deterministic offline rule parser
        lower = text.lower()

        if any(lower.startswith(w) or f" {w} " in lower for w in ["save", "create", "add note", "remember", "write down"]):
            # Extract title / body heuristic
            title = text
            body = text
            if "about" in text:
                parts = text.split("about", 1)
                title = parts[1].split("—")[0].split("-")[0].strip()
                body = parts[1].strip()
            elif ":" in text:
                parts = text.split(":", 1)
                title = parts[0].replace("save", "").replace("add note", "").strip()
                body = parts[1].strip()
            
            tags = []
            if "tag" in text.lower():
                tag_match = re.findall(r"tag(?:ged)?\s+(?:as\s+)?(\w+)", text, re.IGNORECASE)
                if tag_match:
                    tags = tag_match

            return {"intent": "add", "args": {"title": title.capitalize(), "body": body, "tags": tags}}

        elif any(w in lower for w in ["search", "find", "list", "what did i write", "show me", "get notes"]):
            query = text.replace("search", "").replace("find", "").replace("what did i write about", "").strip()
            return {"intent": "search", "args": {"query": query}}

        elif any(w in lower for w in ["update", "modify", "change", "edit"]):
            return {"intent": "update", "args": {"query": text}}

        elif any(w in lower for w in ["delete", "remove", "erase"]):
            return {"intent": "delete", "args": {"query": text}}

        return {"intent": "unknown", "args": {}}
