"""
test_vector_store.py — Tests for VectorStore (Qdrant integration).

These tests require a real GEMINI_API_KEY and QDRANT_URL to run.
They are skipped automatically if the environment variables are not set.

To run these tests: make sure your .env file has valid API keys.
"""

import os
import pytest
from models import Note

# Skip all tests in this file if GEMINI_API_KEY is not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not os.getenv("QDRANT_URL"),
    reason="GEMINI_API_KEY and QDRANT_URL required for vector store tests",
)


def test_vector_store_upsert_and_search():
    """
    Save two notes with different topics, then verify semantic search
    returns the correct note for each topic-specific query.
    """
    from vector_store import VectorStore
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

    # Cleanup
    vstore.delete_note(note1.id)
    vstore.delete_note(note2.id)


def test_vector_store_delete():
    """
    Save a note, verify it's searchable, delete it, verify it's gone.
    """
    from vector_store import VectorStore
    vstore = VectorStore()

    note = Note(title="Temporary Note", body="This note will be deleted from Qdrant")
    vstore.upsert_note(note)

    results_before = vstore.search("temporary note delete qdrant")
    assert note.id in results_before

    vstore.delete_note(note.id)

    results_after = vstore.search("temporary note delete qdrant")
    assert note.id not in results_after
