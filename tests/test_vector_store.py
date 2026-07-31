"""
test_vector_store.py — Tests for VectorStore (Qdrant integration).
"""

import os
import pytest
from src.models import Note

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not os.getenv("QDRANT_URL"),
    reason="GEMINI_API_KEY and QDRANT_URL required for vector store tests",
)


def test_vector_store_upsert_and_search():
    from src.vector_store import VectorStore
    vstore = VectorStore()

    note1 = Note(title="Python Asyncio", body="Tutorial on python asyncio and event loops", tags=["python"])
    note2 = Note(title="Pasta Recipe", body="Boil water, add salt, cook pasta with tomato sauce", tags=["cooking"])

    vstore.upsert_note(note1)
    vstore.upsert_note(note2)

    results_python = vstore.search("asyncio event loop python")
    assert len(results_python) > 0
    assert results_python[0] == note1.id

    results_pasta = vstore.search("pasta tomato sauce recipe")
    assert len(results_pasta) > 0
    assert results_pasta[0] == note2.id

    vstore.delete_note(note1.id)
    vstore.delete_note(note2.id)


def test_vector_store_delete():
    from src.vector_store import VectorStore
    vstore = VectorStore()

    note = Note(title="Temporary Note", body="This note will be deleted from Qdrant")
    vstore.upsert_note(note)

    results_before = vstore.search("temporary note delete qdrant")
    assert note.id in results_before

    vstore.delete_note(note.id)

    results_after = vstore.search("temporary note delete qdrant")
    assert note.id not in results_after
