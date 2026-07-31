"""
llm.py — All LLM interactions in one place.

Two functions, one per LLM call:

  1. pick_intent(user_message, history) → IntentResult
     Reads user message + conversation history.
     Returns a typed IntentResult — the agent uses it to decide what to do.

  2. rewrite_note(old_note, user_instruction) → RewrittenNote
     Receives the existing note + what the user wants to change.
     Returns a typed RewrittenNote with all updated fields.

Both functions use LangChain structured output:
  llm.with_structured_output(Schema) forces the model to return data
  that matches the Pydantic schema — no JSON parsing, no string cleaning.

Output shapes are defined in schemas.py.
No other file calls the Gemini LLM — everything goes through here.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

import config
from schemas import IntentResult, RewrittenNote


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create and return a Gemini LLM instance using settings from config.py."""
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,  # temperature=0 → deterministic, no randomness
    )


def pick_intent(user_message: str, history: list[dict]) -> IntentResult:
    """
    Decide what the user wants to do and extract the relevant arguments.

    Sends the last MEMORY_SIZE messages from conversation history + the current
    user message to the LLM. Returns a typed IntentResult.

    The LLM fills:
      - intent  → one of: save, search_keyword, search_tags, search_semantic,
                          update, delete, unknown
      - the relevant fields for that intent (title/body/tags, keyword, or query)

    Because we use structured output, the response is already a valid
    IntentResult object — no parsing needed.
    """
    # Build a readable conversation history string to give the LLM context
    history_lines = []
    for msg in history[-config.MEMORY_SIZE:]:
        role = msg["role"].capitalize()   # "user" → "User", "assistant" → "Assistant"
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines) if history_lines else "(no history yet)"

    prompt = f"""You are a note-taking assistant. Analyze the conversation and decide what the user wants to do.

Conversation history:
{history_text}

Current user message: "{user_message}"

Choose the correct intent and fill in the relevant fields:

- save: user wants to create a new note
  → fill: title (short, 3-8 words), body (clean note content), tags (1-3 categories)

- search_keyword: user is looking for notes that contain a specific word
  → fill: keyword

- search_tags: user wants to filter notes by category/tag
  → fill: tags

- search_semantic: user asks a natural language question about their notes
  → fill: query

- update: user wants to change an existing note
  → fill: query (describe what note to find and what to change)

- delete: user wants to remove a note
  → fill: query (describe which note to delete)

- unknown: intent is unclear"""

    # with_structured_output(IntentResult) forces the LLM to return
    # a valid IntentResult object — the agent can use it directly
    llm = _get_llm().with_structured_output(IntentResult)

    try:
        return llm.invoke(prompt)
    except Exception as e:
        print(f"[Warning] LLM intent parsing failed: {e}")
        return IntentResult(intent="unknown")   # safe fallback


def rewrite_note(old_note, user_instruction: str) -> RewrittenNote:
    """
    Rewrite an existing note based on the user's update instruction.

    Sends the full old note (title, body, tags) + the user's instruction
    to the LLM. The LLM applies ONLY the requested change and keeps
    everything else exactly the same.

    Returns a typed RewrittenNote — the agent can use it directly.

    Example:
      old_note.body    = "Standup is every Tuesday at 10am."
      user_instruction = "Change it to Wednesdays"
      → RewrittenNote.body = "Standup is every Wednesday at 10am."
    """
    prompt = f"""Update this note based on the user's instruction. Apply ONLY the requested change — keep everything else exactly the same.

Current note:
  Title: {old_note.title}
  Body:  {old_note.body}
  Tags:  {", ".join(old_note.tags) if old_note.tags else "(none)"}

User's instruction: "{user_instruction}" """

    # with_structured_output(RewrittenNote) forces the LLM to return
    # a valid RewrittenNote object — no parsing needed
    llm = _get_llm().with_structured_output(RewrittenNote)

    try:
        return llm.invoke(prompt)
    except Exception as e:
        print(f"[Warning] LLM note rewrite failed: {e}")
        # Safe fallback: return the old note unchanged
        return RewrittenNote(
            title=old_note.title,
            body=old_note.body,
            tags=old_note.tags,
        )
