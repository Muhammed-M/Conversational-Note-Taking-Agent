"""
llm.py — All LLM interactions in one place.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.schemas import IntentResult, RewrittenNote


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create and return a Gemini LLM instance using settings from config.py."""
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
    )


def pick_intent(user_message: str, history: list[dict]) -> IntentResult:
    """
    Decide what the user wants to do and extract relevant arguments.
    Returns a typed IntentResult.
    """
    history_lines = []
    for msg in history[-config.MEMORY_SIZE:]:
        role = msg["role"].capitalize()
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines) if history_lines else "(no history yet)"

    prompt = f"""You are a helpful conversational note-taking assistant. Analyze the conversation and decide what the user wants to do.

Conversation history:
{history_text}

Current user message: "{user_message}"

Choose the correct intent and fill in the relevant fields:

- save: user wants to create or save a new note
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

- chitchat: user is greeting, saying thanks, asking who you are, or making casual conversation (e.g. "hi", "hello", "thank you", "thanks!", "how are you?")
  → fill: chitchat_response (write a warm, friendly, natural response acknowledging what they said and offering help with their notes)

- unknown: intent is completely unclear"""

    print(f"\n[TRACKING] 🧠 Intent Router -> sending message to LLM: '{user_message}'")
    llm = _get_llm().with_structured_output(IntentResult)

    try:
        res = llm.invoke(prompt)
        print(f"[TRACKING] 🎯 Intent Router output -> intent='{res.intent}'")
        return res
    except Exception as e:
        print(f"[TRACKING] ⚠️ LLM intent parsing failed: {e}")
        return IntentResult(intent="unknown")


def rewrite_note(old_note, user_instruction: str) -> RewrittenNote:
    """
    Rewrite an existing note based on user instruction.
    Returns a typed RewrittenNote.
    """
    print(f"\n[TRACKING] ✍️ LLM Rewrite -> updating note '{old_note.title}' with instruction: '{user_instruction}'")
    prompt = f"""Update this note based on the user's instruction. Apply ONLY the requested change — keep everything else exactly the same.

Current note:
  Title: {old_note.title}
  Body:  {old_note.body}
  Tags:  {", ".join(old_note.tags) if old_note.tags else "(none)"}

User's instruction: "{user_instruction}" """

    llm = _get_llm().with_structured_output(RewrittenNote)

    try:
        res = llm.invoke(prompt)
        print(f"[TRACKING] ✅ LLM Rewrite finished -> new title: '{res.title}'")
        return res
    except Exception as e:
        print(f"[TRACKING] ⚠️ LLM note rewrite failed: {e}")
        return RewrittenNote(
            title=old_note.title,
            body=old_note.body,
            tags=old_note.tags,
        )
