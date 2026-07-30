"""
Domain Data Models for Note-Taking Agent.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Note:
    """
    Represents a single note entity in the store.
    
    Attributes:
        id: Unique identifier (UUID4 string).
        title: Short title summarizing the note.
        body: Main text content of the note.
        tags: List of text tags associated with the note.
        created_at: ISO 8601 timestamp string of creation.
        updated_at: ISO 8601 timestamp string of last modification.
    """
    title: str
    body: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def short_id(self) -> str:
        """Return short 8-character ID prefix for user-facing CLI references."""
        return self.id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert Note instance to a dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Note":
        """Construct Note instance from dictionary representation."""
        tags = data.get("tags", [])
        if isinstance(tags, str):
            # Fallback if tags was JSON stringified or comma separated
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            body=data.get("body", ""),
            tags=list(tags),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )
