"""
test_store.py — Unit tests for NoteStore (SQLite layer).

Tests all CRUD operations:
  - add and retrieve a note
  - update a note
  - delete a note
  - search by keyword
  - search by tags
"""

import os
import shutil
import tempfile
import pytest
from models import Note
from store import NoteStore


@pytest.fixture
def store():
    """
    Create a temporary SQLite database for each test.
    The database is deleted automatically after the test finishes.
    We use shutil.rmtree with ignore_errors so Windows file locks don't fail teardown.
    """
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_notes.db")
    store_inst = NoteStore(db_path=db_path)
    yield store_inst
    # Cleanup (ignore_errors handles Windows SQLite file locks)
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_and_get_note(store):
    """Save a note and retrieve it by full ID and by short 8-char ID."""
    note = store.add_note(title="Team Standup", body="Standup is every Tuesday", tags=["meetings"])

    assert note.id is not None
    assert note.title == "Team Standup"
    assert note.tags == ["meetings"]

    # Retrieve by full UUID
    retrieved = store.get_note_by_id(note.id)
    assert retrieved is not None
    assert retrieved.title == "Team Standup"

    # Retrieve by short 8-character prefix
    retrieved_short = store.get_note_by_id(note.short_id)
    assert retrieved_short is not None
    assert retrieved_short.id == note.id


def test_update_note(store):
    """Update a note's title, body, and tags. Verify updated_at changes."""
    note = store.add_note(title="Old Title", body="Old body text", tags=["old"])

    updated = store.update_note(
        note_id=note.id,
        title="New Title",
        body="New body text",
        tags=["new"],
    )

    assert updated is not None
    assert updated.title == "New Title"
    assert updated.body == "New body text"
    assert updated.tags == ["new"]
    assert updated.created_at == note.created_at  # created_at must not change
    assert updated.updated_at >= note.updated_at   # updated_at must be refreshed


def test_delete_note(store):
    """Delete a note and verify it can no longer be found."""
    note = store.add_note(title="To Delete", body="Temporary content")

    assert store.delete_note(note.id) is True    # first delete returns True
    assert store.get_note_by_id(note.id) is None  # note is gone
    assert store.delete_note(note.id) is False   # second delete returns False (already gone)


def test_search_by_keyword(store):
    """Search notes by keyword. Only matching notes should be returned."""
    n1 = store.add_note(title="Python API design", body="REST guidelines", tags=["backend"])
    n2 = store.add_note(title="Frontend UI", body="React component setup", tags=["frontend"])
    n3 = store.add_note(title="Office relocation", body="New address in London", tags=["personal"])

    results = store.search_by_keyword("API")
    assert len(results) == 1
    assert results[0].id == n1.id

    results_none = store.search_by_keyword("nonexistent_xyz_12345")
    assert len(results_none) == 0


def test_search_by_tags(store):
    """Search notes by tag. Only notes with matching tags should be returned."""
    n1 = store.add_note(title="Python Tutorial", body="Learn Python basics", tags=["python", "learning"])
    n2 = store.add_note(title="Pasta Recipe", body="Boil water, add pasta", tags=["cooking"])

    results = store.search_by_tags(["python"])
    assert len(results) == 1
    assert results[0].id == n1.id

    results_cooking = store.search_by_tags(["cooking"])
    assert len(results_cooking) == 1
    assert results_cooking[0].id == n2.id

    results_none = store.search_by_tags(["nonexistent_tag"])
    assert len(results_none) == 0


def test_get_note_not_found(store):
    """Fetching a non-existent note returns None."""
    result = store.get_note_by_id("00000000-0000-0000-0000-000000000000")
    assert result is None
