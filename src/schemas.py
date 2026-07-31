"""
schemas.py — Pydantic schemas for all LLM input/output shapes in this project.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """
    Output schema for pick_intent().
    """

    intent: Literal[
        "save",
        "search_keyword",
        "search_tags",
        "search_semantic",
        "update",
        "delete",
        "unknown",
    ] = Field(description="What the user wants to do.")

    title: Optional[str] = Field(
        default=None,
        description="Short note title (3-8 words). LLM generates this if the user didn't provide one.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Full note body text, written cleanly from the user's message.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="List of 1-3 category tags (e.g. ['work', 'meetings']). Used for saving and tag search.",
    )
    keyword: Optional[str] = Field(
        default=None,
        description="The specific word to search for in note titles and bodies.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Natural language description — what to find, what to change, or which note to delete.",
    )


class RewrittenNote(BaseModel):
    """
    Output schema for rewrite_note().
    """

    title: str = Field(description="The updated note title (keep the same if title wasn't changed).")
    body: str = Field(description="The updated note body with the requested change applied.")
    tags: list[str] = Field(description="The updated list of tags (keep the same if tags weren't changed).")
