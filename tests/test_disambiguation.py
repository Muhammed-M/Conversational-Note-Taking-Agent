"""
Unit and integration tests for Intent Disambiguation Gate.
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
    db_path = os.path.join(tmpdir, "test_disambig.db")

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


def test_disambiguation_flow(graph_app):
    graph, store = graph_app

    # Create 2 similar notes
    store.add_note(title="Project Alpha Review", body="Quarterly review for Alpha", tags=["work"])
    store.add_note(title="Project Beta Review", body="Quarterly review for Beta", tags=["work"])

    state = create_initial_state()

    # User attempts ambiguous delete request
    state = graph.run("Delete the project review note", state)

    # 1. State must enter AWAITING_DISAMBIGUATION
    assert state["mode"] == "AWAITING_DISAMBIGUATION"
    assert len(state["search_candidates"]) == 2
    assert "Multiple notes matched" in state["final_response"]

    # 2. User selects option 1
    state = graph.run("1", state)

    # 3. State transitions to AWAITING_CONFIRM for option 1
    assert state["mode"] == "AWAITING_CONFIRM"
    assert state["pending_action"]["note_id"] == state["search_candidates"][0]["id"]
    assert "Are you sure" in state["final_response"]
