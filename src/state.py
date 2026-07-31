"""
AgentState Definition for Note-Taking Agent.
"""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared conversation state passed through agent calls.
    """
    messages: list[dict[str, Any]]
    pending_action: Optional[dict[str, Any]]
    last_note_id: Optional[str]
    search_candidates: Optional[list[dict[str, Any]]]
    mode: str
    final_response: Optional[str]


def create_initial_state() -> AgentState:
    """Helper function to create a clean default state."""
    return {
        "messages": [],
        "pending_action": None,
        "last_note_id": None,
        "search_candidates": None,
        "mode": "IDLE",
        "final_response": None,
    }
