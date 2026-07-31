"""
store.py — SQLite storage for notes.

This module handles all reading and writing of notes to the SQLite database.
It provides these operations:
  - add_note       → save a new note
  - get_note_by_id → fetch a note by its ID
  - search_by_keyword → find notes that contain a specific word
  - search_by_tags    → find notes that have specific tags
  - update_note    → change an existing note
  - delete_note    → remove a note

The vector store (Qdrant) is kept in sync by the agent — not here.
This file only deals with SQLite.
"""

import json
import sqlite3
from typing import Optional

from models import Note, utc_now_iso
import config


class NoteStore:
    """SQLite database store for notes."""

    def __init__(self, db_path: str = "notes.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Open a connection to the SQLite database file."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # makes rows accessible by column name (e.g. row["title"])
        return conn

    def _init_db(self) -> None:
        """Create the notes table if it does not already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id         TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    body       TEXT NOT NULL,
                    tags       TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _row_to_note(self, row: sqlite3.Row) -> Note:
        """Convert a raw SQLite row into a Note object."""
        return Note(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            tags=json.loads(row["tags"]),  # tags are stored as a JSON string like '["work", "meetings"]'
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Create ────────────────────────────────────────────────────────────────

    def add_note(self, title: str, body: str, tags: list[str] = None) -> Note:
        """
        Save a new note to SQLite and return it.
        The Note model auto-generates a UUID and timestamps.
        """
        note = Note(title=title, body=body, tags=tags or [])

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO notes (id, title, body, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (note.id, note.title, note.body, json.dumps(note.tags), note.created_at, note.updated_at),
            )
            conn.commit()

        return note

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_note_by_id(self, note_id: str) -> Optional[Note]:
        """
        Fetch a note by its full UUID or its short 8-character ID prefix.
        Returns None if not found.
        """
        with self._get_connection() as conn:
            # Try exact match first (full UUID)
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()

            # If not found and the ID looks short, try prefix match
            if not row and len(note_id) < 36:
                row = conn.execute("SELECT * FROM notes WHERE id LIKE ?", (f"{note_id}%",)).fetchone()

        return self._row_to_note(row) if row else None

    def search_by_keyword(self, keyword: str, top_n: int = None) -> list[Note]:
        """
        Find notes where the title, body, or tags contain the given keyword.
        Uses SQL LIKE for a case-insensitive partial match.
        Returns at most top_n notes, ordered by most recently updated.
        """
        top_n = top_n or config.TOP_K_KEYWORD
        kw = f"%{keyword.strip()}%"  # wrap in % wildcards for LIKE matching

        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM notes
                   WHERE title LIKE ? OR body LIKE ? OR tags LIKE ?
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (kw, kw, kw, top_n),
            ).fetchall()

        return [self._row_to_note(row) for row in rows]

    def search_by_tags(self, tags: list[str], top_n: int = None) -> list[Note]:
        """
        Find notes that have at least one of the given tags.
        Loads all notes and filters in memory (fine for a small local database).
        Returns at most top_n notes, ordered by most recently updated.
        """
        top_n = top_n or config.TOP_K_TAG

        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()

        results = []
        for row in rows:
            note_tags = [t.lower() for t in json.loads(row["tags"])]
            # Check if any of the user's requested tags appear in this note's tags
            if any(tag.lower() in note_tags for tag in tags):
                results.append(self._row_to_note(row))
            if len(results) >= top_n:
                break

        return results

    # ── Update ────────────────────────────────────────────────────────────────

    def update_note(self, note_id: str, title: str, body: str, tags: list[str]) -> Optional[Note]:
        """
        Update all fields of an existing note.
        Sets a new updated_at timestamp.
        Returns the updated Note, or None if the note was not found.
        """
        existing = self.get_note_by_id(note_id)
        if not existing:
            return None

        new_updated_at = utc_now_iso()

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE notes SET title = ?, body = ?, tags = ?, updated_at = ? WHERE id = ?",
                (title, body, json.dumps(tags), new_updated_at, existing.id),
            )
            conn.commit()

        return Note(
            id=existing.id,
            title=title,
            body=body,
            tags=tags,
            created_at=existing.created_at,  # created_at never changes
            updated_at=new_updated_at,
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note by ID.
        Returns True if the note was found and deleted, False if it didn't exist.
        """
        existing = self.get_note_by_id(note_id)
        if not existing:
            return False

        with self._get_connection() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (existing.id,))
            conn.commit()

        return True
