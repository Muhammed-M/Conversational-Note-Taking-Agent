"""
Strictly typed Pydantic tool schemas for the LLM function calling interface.

These schemas act as the contract between the conversational agent (Gemini)
and the underlying storage repository.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AddNoteInput(BaseModel):
    """Schema for adding a new note."""
    title: str = Field(description="The headline title of the note.")
    body: str = Field(description="The detailed content/body of the note.")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags or categories (e.g. ['meetings', 'work']).",
    )


class SearchNotesInput(BaseModel):
    """Schema for querying or searching notes."""
    query: Optional[str] = Field(
        default=None,
        description="Natural language query, topic, or keyword to search for.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional list of tags to filter by (e.g. ['meetings']).",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="ISO date string filter start range (e.g. '2026-07-01').",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="ISO date string filter end range.",
    )


class UpdateNoteInput(BaseModel):
    """Schema for updating an existing note."""
    note_id: str = Field(
        description="Unique ID or 8-character prefix of the target note."
    )
    title: Optional[str] = Field(
        default=None, description="New title for the note if updating."
    )
    body: Optional[str] = Field(
        default=None, description="New body text for the note if updating."
    )
    tags: Optional[list[str]] = Field(
        default=None, description="New list of tags for the note if updating."
    )


class DeleteNoteInput(BaseModel):
    """Schema for deleting a note."""
    note_id: str = Field(
        description="Unique ID or 8-character prefix of the note to delete."
    )
