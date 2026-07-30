"""
SQLite Canonical Storage Layer for Notes.

All canonical CRUD operations happen here first. This layer handles structured
filtering (tags, date ranges) via SQL, and provides callback hooks to keep
the vector search index in sync.
"""

import json
import sqlite3
from typing import Callable, Optional
from models import Note, utc_now_iso


# Sync callback type definitions
OnMutationCallback = Callable[[Note], None]
OnDeleteCallback = Callable[[str], None]


class NoteStore:
    """
    SQLite repository for Note persistent storage.
    """

    def __init__(self, db_path: str = "notes.db"):
        self.db_path = db_path
        self._on_created_listeners: list[OnMutationCallback] = []
        self._on_updated_listeners: list[OnMutationCallback] = []
        self._on_deleted_listeners: list[OnDeleteCallback] = []
        self.init_db()

    def register_on_created(self, callback: OnMutationCallback) -> None:
        """Register a callback to run after a note is created."""
        self._on_created_listeners.append(callback)

    def register_on_updated(self, callback: OnMutationCallback) -> None:
        """Register a callback to run after a note is updated."""
        self._on_updated_listeners.append(callback)

    def register_on_deleted(self, callback: OnDeleteCallback) -> None:
        """Register a callback to run after a note is deleted."""
        self._on_deleted_listeners.append(callback)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize the SQLite database table structure if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_note(self, title: str, body: str, tags: Optional[list[str]] = None) -> Note:
        """
        Create and persist a new note in SQLite.
        Triggers vector index sync callbacks.
        """
        tags_list = tags if tags is not None else []
        note = Note(title=title, body=body, tags=tags_list)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO notes (id, title, body, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.title,
                    note.body,
                    json.dumps(note.tags),
                    note.created_at,
                    note.updated_at,
                ),
            )
            conn.commit()

        # Trigger registered vector store listeners
        for listener in self._on_created_listeners:
            try:
                listener(note)
            except Exception as e:
                print(f"[Warning] Note store on_created listener failed: {e}")

        return note

    def get_note_by_id(self, note_id: str) -> Optional[Note]:
        """
        Retrieve a note by exact UUID or short 8-char ID prefix.
        Returns None if no matching note is found.
        """
        with self._get_connection() as conn:
            # First try exact match
            cursor = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            
            # Fall back to prefix match if short ID was supplied
            if not row and len(note_id) < 36:
                cursor = conn.execute("SELECT * FROM notes WHERE id LIKE ?", (f"{note_id}%",))
                row = cursor.fetchone()

            if not row:
                return None

            return Note(
                id=row["id"],
                title=row["title"],
                body=row["body"],
                tags=json.loads(row["tags"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def search_notes_sql(
        self,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[Note]:
        """
        Perform structured filtering and SQL keyword search on SQLite.
        Used directly or as fallback when vector search is unavailable.
        """
        conditions = []
        params: list[str] = []

        if query and query.strip():
            kw = f"%{query.strip()}%"
            conditions.append("(title LIKE ? OR body LIKE ? OR tags LIKE ?)")
            params.extend([kw, kw, kw])

        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM notes{where_clause} ORDER BY updated_at DESC"

        notes: list[Note] = []
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            for row in rows:
                parsed_tags = json.loads(row["tags"])
                # If specific tag filters were requested, verify match in memory
                if tags:
                    normal_tags = [t.lower() for t in parsed_tags]
                    if not any(t.lower() in normal_tags for t in tags):
                        continue
                notes.append(
                    Note(
                        id=row["id"],
                        title=row["title"],
                        body=row["body"],
                        tags=parsed_tags,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )

        return notes

    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        body: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[Note]:
        """
        Update an existing note in SQLite and notify listeners.
        """
        existing = self.get_note_by_id(note_id)
        if not existing:
            return None

        new_title = title if title is not None else existing.title
        new_body = body if body is not None else existing.body
        new_tags = tags if tags is not None else existing.tags
        updated_at = utc_now_iso()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE notes
                SET title = ?, body = ?, tags = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_body, json.dumps(new_tags), updated_at, existing.id),
            )
            conn.commit()

        updated_note = Note(
            id=existing.id,
            title=new_title,
            body=new_body,
            tags=new_tags,
            created_at=existing.created_at,
            updated_at=updated_at,
        )

        # Trigger listeners
        for listener in self._on_updated_listeners:
            try:
                listener(updated_note)
            except Exception as e:
                print(f"[Warning] Note store on_updated listener failed: {e}")

        return updated_note

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note by ID from SQLite and notify listeners.
        Returns True if deleted, False if note was not found.
        """
        existing = self.get_note_by_id(note_id)
        if not existing:
            return False

        with self._get_connection() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (existing.id,))
            conn.commit()

        # Trigger listeners
        for listener in self._on_deleted_listeners:
            try:
                listener(existing.id)
            except Exception as e:
                print(f"[Warning] Note store on_deleted listener failed: {e}")

        return True
