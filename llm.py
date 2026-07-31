"""
llm.py — All LLM interactions in one place.

This module has two functions:

  1. pick_intent(user_message, history)
     Reads the user message + recent conversation history.
     Asks Gemini: "what does the user want to do?"
     Returns a dict with 'intent' and 'args'.

  2. rewrite_note(old_note, user_instruction)
     Receives the full existing note + what the user wants to change.
     Asks Gemini to apply only the requested change and return the updated note.
     Returns a dict with 'title', 'body', 'tags'.

No other file calls the Gemini LLM — everything goes through here.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
import config


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create and return a Gemini LLM instance using settings from config.py."""
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,  # temperature=0 means deterministic, no randomness
    )


def _parse_json(text: str) -> dict:
    """
    Parse JSON from LLM response text.
    The LLM sometimes wraps output in markdown code fences like ```json ... ```
    This function strips those before parsing.
    """
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If we still can't parse it, return an empty/unknown result
        return {}


def pick_intent(user_message: str, history: list[dict]) -> dict:
    """
    Decide what the user wants to do and extract the relevant arguments.

    Sends the last MEMORY_SIZE messages from conversation history + the current
    user message to Gemini. The LLM returns a JSON object with:
      - "intent": one of "save", "search_keyword", "search_tags",
                  "search_semantic", "update", "delete", or "unknown"
      - "args": a dict with the fields relevant to that intent

    Intent → Args mapping:
      save            → { title, body, tags }        LLM generates these from the message
      search_keyword  → { keyword }                  a specific word to search for in notes
      search_tags     → { tags: [] }                 specific tag names to filter by
      search_semantic → { query }                    a natural language question
      update          → { query }                    description of what to update
      delete          → { query }                    description of which note to delete
      unknown         → {}
    """
    # Build a readable history string from the last N messages
    history_lines = []
    for msg in history[-config.MEMORY_SIZE:]:
        role = msg["role"].capitalize()  # "user" → "User", "assistant" → "Assistant"
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines) if history_lines else "(no history yet)"

    prompt = f"""You are a note-taking assistant. Based on the conversation history and the latest user message, decide what the user wants to do.

Conversation history:
{history_text}

Current user message: "{user_message}"

Choose the correct intent and extract the relevant arguments. Return a JSON object.

Intent options and their required args:
- "save"            → args: {{ "title": "...", "body": "...", "tags": ["..."] }}
  Use this when the user wants to save or create a new note.
  Generate a short title (3-8 words), clean body text, and 1-3 relevant tags from the message.

- "search_keyword"  → args: {{ "keyword": "..." }}
  Use this when the user is looking for notes that contain a specific word.

- "search_tags"     → args: {{ "tags": ["..."] }}
  Use this when the user mentions specific tags or categories to filter by.

- "search_semantic" → args: {{ "query": "..." }}
  Use this when the user asks a natural language question about their notes.

- "update"          → args: {{ "query": "..." }}
  Use this when the user wants to change or edit an existing note.

- "delete"          → args: {{ "query": "..." }}
  Use this when the user wants to remove or delete a note.

- "unknown"         → args: {{}}
  Use this if the intent is unclear.

Return ONLY the raw JSON object. No markdown, no explanation.
Example: {{"intent": "save", "args": {{"title": "Team Standup", "body": "Standup is every Tuesday at 10am.", "tags": ["meetings", "work"]}}}}"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    result = _parse_json(response.content)

    # Make sure the result has the expected structure
    if "intent" not in result:
        return {"intent": "unknown", "args": {}}

    if "args" not in result:
        result["args"] = {}

    return result


def rewrite_note(old_note, user_instruction: str) -> dict:
    """
    Rewrite an existing note based on what the user wants to change.

    Sends the full old note (title, body, tags) + the user's update instruction to Gemini.
    The LLM applies ONLY the requested change and keeps everything else the same.

    Returns: { "title": "...", "body": "...", "tags": ["..."] }

    Example:
      old_note.body = "Standup is every Tuesday at 10am."
      user_instruction = "Change it to Wednesdays"
      → returns body = "Standup is every Wednesday at 10am."
    """
    prompt = f"""The user wants to update an existing note. Apply ONLY the change they requested and keep everything else exactly the same.

Current note:
  Title: {old_note.title}
  Body:  {old_note.body}
  Tags:  {", ".join(old_note.tags) if old_note.tags else "(none)"}

User's update instruction: "{user_instruction}"

Return the updated note as a raw JSON object:
{{"title": "...", "body": "...", "tags": ["..."]}}

Return ONLY the raw JSON. No explanation."""

    llm = _get_llm()
    response = llm.invoke(prompt)
    result = _parse_json(response.content)

    # If parsing failed, return the old note unchanged as a safe fallback
    if not result:
        return {
            "title": old_note.title,
            "body": old_note.body,
            "tags": old_note.tags,
        }

    return result
