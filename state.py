"""
AgentState Definition for LangGraph State Machine.
"""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared conversation state passed through the LangGraph graph nodes.
    
    Attributes:
        messages: List of chat messages in history.
        pending_action: Draft payload for action requiring user confirmation/disambiguation.
        last_note_id: ID of the last created, retrieved, or modified note (for multi-turn anaphora).
        search_candidates: List of candidate notes awaiting user disambiguation.
        mode: Operational state ("IDLE", "AWAITING_CONFIRM", "AWAITING_DISAMBIGUATION").
        final_response: String response ready to display to the user.
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
