"""
schemas.py — Pydantic schemas for all LLM input/output shapes in this project.

Why this file exists:
  Every LLM call has a defined output shape. Instead of hoping the LLM returns
  valid JSON and manually parsing it, we define the exact shape here using Pydantic.
  LangChain's structured output then forces the LLM to match this shape —
  no JSON parsing, no string cleaning, no surprises.

This file is the single place to look when you want to know:
  "what does this LLM call return?"

Two schemas are defined here, one per LLM call:
  IntentResult  → output of pick_intent()  in llm.py
  RewrittenNote → output of rewrite_note() in llm.py
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """
    Output schema for pick_intent().

    The LLM reads the user's message and conversation history,
    then fills in this schema to tell the agent what the user wants.

    The 'intent' field is the routing decision.
    The other fields are the arguments for that intent.

    Fields are FLAT (not nested) — simple to read and access in agent.py.
    Each intent only uses the fields it needs; the rest stay None.

    Intent → which fields are filled:
      save            → title, body, tags
      search_keyword  → keyword
      search_tags     → tags
      search_semantic → query
      update          → query
      delete          → query
      unknown         → (none)
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

    # ── Used when intent = 'save' ──────────────────────────────────────────────
    title: Optional[str] = Field(
        default=None,
        description="Short note title (3-8 words). LLM generates this if the user didn't provide one.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Full note body text, written cleanly from the user's message.",
    )

    # ── Used when intent = 'save' or 'search_tags' ────────────────────────────
    tags: Optional[list[str]] = Field(
        default=None,
        description="List of 1-3 category tags (e.g. ['work', 'meetings']). Used for saving and tag search.",
    )

    # ── Used when intent = 'search_keyword' ───────────────────────────────────
    keyword: Optional[str] = Field(
        default=None,
        description="The specific word to search for in note titles and bodies.",
    )

    # ── Used when intent = 'search_semantic', 'update', or 'delete' ──────────
    query: Optional[str] = Field(
        default=None,
        description="Natural language description — what to find, what to change, or which note to delete.",
    )


class RewrittenNote(BaseModel):
    """
    Output schema for rewrite_note().

    The LLM receives the old note + the user's update instruction,
    applies only the requested change, and returns this schema
    with the fully updated note fields.

    Example:
      Old body: "Standup is every Tuesday at 10am."
      Instruction: "Change it to Wednesdays"
      New body: "Standup is every Wednesday at 10am."
    """

    title: str = Field(description="The updated note title (keep the same if title wasn't changed).")
    body: str = Field(description="The updated note body with the requested change applied.")
    tags: list[str] = Field(description="The updated list of tags (keep the same if tags weren't changed).")
