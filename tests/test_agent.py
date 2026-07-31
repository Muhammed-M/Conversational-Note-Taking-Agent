"""
test_agent.py — Integration tests for the Agent conversation flows.
"""

import os
import shutil
import tempfile
import pytest

from src.agent import Agent
from src.schemas import IntentResult
from src.state import create_initial_state
from src.store import NoteStore
from src.models import Note


class MockVectorStore:
    """Fake VectorStore that does nothing for unit tests."""

    def upsert_note(self, note: Note) -> None:
        pass

    def delete_note(self, note_id: str) -> None:
        pass

    def search(self, query: str, top_k: int = 1) -> list[str]:
        return []


@pytest.fixture
def app():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    store = NoteStore(db_path=db_path)
    vector_store = MockVectorStore()
    agent = Agent(store=store, vector_store=vector_store)
    yield agent, store
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_flow(app):
    agent, store = app
    state = create_initial_state()

    result = IntentResult(
        intent="save",
        title="Team Standup",
        body="Standup is every Tuesday at 10am.",
        tags=["meetings"],
    )
    state = agent._save_note(result, state)

    assert "Saved" in state["final_response"] or "✅" in state["final_response"]
    assert state["last_note_id"] is not None

    note = store.get_note_by_id(state["last_note_id"])
    assert note is not None
    assert note.title == "Team Standup"


def test_keyword_search_flow(app):
    agent, store = app
    store.add_note(title="Python Tips", body="Use list comprehensions in Python", tags=["python"])

    state = create_initial_state()
    state = agent._search_by_keyword(IntentResult(intent="search_keyword", keyword="Python"), state)

    assert "Python Tips" in state["final_response"]


def test_tag_search_flow(app):
    agent, store = app
    store.add_note(title="Meeting Notes", body="Q3 roadmap discussed", tags=["meetings", "work"])

    state = create_initial_state()
    state = agent._search_by_tags(IntentResult(intent="search_tags", tags=["meetings"]), state)

    assert "Meeting Notes" in state["final_response"]


def test_delete_cancel_flow(app):
    agent, store = app
    note = store.add_note(title="Important Note", body="Do not delete this")

    state = create_initial_state()
    state["pending_action"] = {"intent": "delete", "note_id": note.id, "note_title": note.title}
    state["mode"] = "AWAITING_CONFIRM"
    state["messages"].append({"role": "user", "content": "no"})

    state = agent._resolve_confirmation("no", state)

    assert state["mode"] == "IDLE"
    assert "cancelled" in state["final_response"].lower()
    assert store.get_note_by_id(note.id) is not None


def test_delete_confirm_flow(app):
    agent, store = app
    note = store.add_note(title="Old Note", body="Temporary content")

    state = create_initial_state()
    state["pending_action"] = {"intent": "delete", "note_id": note.id, "note_title": note.title}
    state["mode"] = "AWAITING_CONFIRM"

    state = agent._resolve_confirmation("yes", state)

    assert state["mode"] == "IDLE"
    assert "Deleted" in state["final_response"] or "🗑️" in state["final_response"]
    assert store.get_note_by_id(note.id) is None


def test_disambiguation_by_number(app):
    agent, store = app
    note1 = store.add_note(title="Alpha Project", body="Alpha details", tags=["work"])
    note2 = store.add_note(title="Beta Project", body="Beta details", tags=["work"])

    state = create_initial_state()
    state["mode"] = "AWAITING_DISAMBIGUATION"
    state["search_candidates"] = [note1.to_dict(), note2.to_dict()]
    state["pending_action"] = {"intent": "delete"}

    state = agent._resolve_disambiguation("1", state)

    assert state["mode"] == "AWAITING_CONFIRM"
    assert state["pending_action"]["note_id"] == note1.id


def test_disambiguation_cancel(app):
    agent, store = app
    note = store.add_note(title="Some Note", body="Some content")

    state = create_initial_state()
    state["mode"] = "AWAITING_DISAMBIGUATION"
    state["search_candidates"] = [note.to_dict()]
    state["pending_action"] = {"intent": "delete"}

    state = agent._resolve_disambiguation("cancel", state)

    assert state["mode"] == "IDLE"
    assert "cancelled" in state["final_response"].lower()
