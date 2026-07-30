"""
Unit tests for VectorNoteStore (Qdrant Vector Store integration).
"""

from models import Note
from vector_store import VectorNoteStore


def test_vector_store_upsert_and_search():
    vstore = VectorNoteStore(location=":memory:", vector_dim=128)

    note1 = Note(title="Python Async Asyncio", body="Tutorial on python asyncio and event loops", tags=["python"])
    note2 = Note(title="Recipe for Pasta", body="Boil water add salt pasta tomato sauce", tags=["cooking"])

    assert vstore.upsert_note(note1) is True
    assert vstore.upsert_note(note2) is True

    # Search query relevant to Python
    res_python = vstore.search("asyncio python loop")
    assert len(res_python) > 0
    assert res_python[0] == note1.id

    # Search query relevant to Pasta
    res_pasta = vstore.search("pasta tomato")
    assert len(res_pasta) > 0
    assert res_pasta[0] == note2.id


def test_vector_store_delete():
    vstore = VectorNoteStore(location=":memory:", vector_dim=128)

    note = Note(title="Temporary Note", body="To be deleted vector")
    vstore.upsert_note(note)

    res_before = vstore.search("Temporary Note")
    assert len(res_before) == 1
    assert res_before[0] == note.id

    assert vstore.delete_note(note.id) is True

    res_after = vstore.search("Temporary Note")
    assert len(res_after) == 0
