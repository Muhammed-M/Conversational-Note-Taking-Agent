"""
Unit tests for NoteStore (SQLite storage layer).
"""

import os
import tempfile
import pytest
from models import Note
from store import NoteStore


@pytest.fixture
def store():
    """Create a temporary SQLite database store for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_notes.db")

    store_inst = NoteStore(db_path=db_path)
    yield store_inst

    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(tmpdir)
    except Exception:
        pass



def test_add_and_get_note(store: NoteStore):
    note = store.add_note(
        title="Team Standup",
        body="Agreed to move standup to Tuesdays",
        tags=["meetings", "work"],
    )

    assert note.id is not None
    assert note.title == "Team Standup"
    assert note.tags == ["meetings", "work"]

    # Retrieve by exact ID
    retrieved = store.get_note_by_id(note.id)
    assert retrieved is not None
    assert retrieved.title == "Team Standup"

    # Retrieve by short 8-char prefix
    retrieved_short = store.get_note_by_id(note.short_id)
    assert retrieved_short is not None
    assert retrieved_short.id == note.id


def test_update_note(store: NoteStore):
    note = store.add_note(title="Old Title", body="Old Body", tags=["old"])
    
    updated = store.update_note(
        note_id=note.id,
        title="New Title",
        body="New Body text",
        tags=["new"],
    )

    assert updated is not None
    assert updated.title == "New Title"
    assert updated.body == "New Body text"
    assert updated.tags == ["new"]
    assert updated.updated_at >= note.updated_at


def test_delete_note(store: NoteStore):
    note = store.add_note(title="To Delete", body="Temporary content")
    
    assert store.delete_note(note.id) is True
    assert store.get_note_by_id(note.id) is None
    assert store.delete_note(note.id) is False  # Second delete returns False


def test_search_notes_sql(store: NoteStore):
    n1 = store.add_note(title="Python API design", body="Rest guidelines", tags=["backend"])
    n2 = store.add_note(title="Frontend UI", body="React component setup", tags=["frontend"])
    n3 = store.add_note(title="Office relocation", body="New address in London", tags=["personal"])

    # Search by keyword
    results_api = store.search_notes_sql(query="API")
    assert len(results_api) == 1
    assert results_api[0].id == n1.id

    # Search by tag
    results_tag = store.search_notes_sql(tags=["frontend"])
    assert len(results_tag) == 1
    assert results_tag[0].id == n2.id

    # Search with no match
    results_none = store.search_notes_sql(query="nonexistent_xyz")
    assert len(results_none) == 0


def test_store_callbacks(store: NoteStore):
    created_events = []
    updated_events = []
    deleted_events = []

    store.register_on_created(lambda n: created_events.append(n.id))
    store.register_on_updated(lambda n: updated_events.append(n.id))
    store.register_on_deleted(lambda nid: deleted_events.append(nid))

    note = store.add_note(title="Event Test", body="Testing callbacks")
    assert len(created_events) == 1
    assert created_events[0] == note.id

    store.update_note(note.id, title="Updated Event Test")
    assert len(updated_events) == 1
    assert updated_events[0] == note.id

    store.delete_note(note.id)
    assert len(deleted_events) == 1
    assert deleted_events[0] == note.id
