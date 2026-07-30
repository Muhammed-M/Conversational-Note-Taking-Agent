"""
Unit and integration tests for Confirmation Gate (cancel & confirm).
"""

import os
import tempfile
import pytest
from graph import NoteAgentGraph
from state import create_initial_state
from store import NoteStore
from vector_store import VectorNoteStore


@pytest.fixture
def graph_app():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_confirm.db")

    store = NoteStore(db_path=db_path)
    vstore = VectorNoteStore(location=":memory:", vector_dim=128)
    graph = NoteAgentGraph(store=store, vector_store=vstore)

    yield graph, store

    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(tmpdir)
    except Exception:
        pass


def test_confirmation_cancellation(graph_app):
    graph, store = graph_app
    note = store.add_note(title="Important Secret Note", body="Do not lose this")

    state = create_initial_state()

    # Trigger delete
    state = graph.run(f"Delete note {note.id}", state)
    assert state["mode"] == "AWAITING_CONFIRM"

    # User cancels with "no"
    state = graph.run("no", state)
    assert state["mode"] == "IDLE"
    assert state["final_response"] == "Action cancelled."
    
    # Verify note was NOT deleted in store
    assert store.get_note_by_id(note.id) is not None


def test_confirmation_success(graph_app):
    graph, store = graph_app
    note = store.add_note(title="Disposable Note", body="Temporary content")

    state = create_initial_state()

    # Trigger delete
    state = graph.run(f"Delete note {note.id}", state)
    assert state["mode"] == "AWAITING_CONFIRM"

    # User confirms with "yes"
    state = graph.run("yes", state)
    assert state["mode"] == "IDLE"
    assert "deleted successfully" in state["final_response"]

    # Verify note IS deleted in store
    assert store.get_note_by_id(note.id) is None


def test_multi_turn_anaphora(graph_app):
    graph, store = graph_app
    
    state = create_initial_state()
    state = graph.run("Save note: Team meeting notes : Discussed Q3 roadmap", state)
    
    assert state["last_note_id"] is not None
    saved_id = state["last_note_id"]

    # Multi-turn follow-up: "Delete that last note"
    state = graph.run("Delete that last note", state)
    assert state["mode"] == "AWAITING_CONFIRM"
    assert state["pending_action"]["note_id"] == saved_id
