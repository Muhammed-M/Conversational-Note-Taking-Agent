# Conversational Note-Taking Agent

A conversational AI agent for managing personal notes through natural language. Understands intent, searches semantically, previews changes, and asks before modifying anything.

Built with **Python**, **LangChain**, **Google Gemini**, **SQLite**, and **Qdrant Cloud**.

---

## How It Works

Every message goes through a **Gemini-powered intent router** that classifies what the user wants, extracts the relevant data (title, keyword, query, etc.), and routes to the right handler — all in a single structured LLM call.

```
User message
     │
     ▼
Intent Router (Gemini + Pydantic structured output)
     │
     ├── save            → write to SQLite + embed in Qdrant
     ├── search_keyword  → SQLite LIKE query
     ├── search_tags     → SQLite tag filter
     ├── search_semantic → Qdrant vector similarity
     ├── update          → find candidates → preview → confirm → write both stores
     ├── delete          → find candidates → confirm → remove from both stores
     ├── chitchat        → inline LLM-generated reply (same call, no extra cost)
     └── unknown         → fixed fallback response
```

**Hybrid storage:** SQLite handles structured queries (keyword, tag). Qdrant handles semantic similarity. Both are written on every save and update.

**Human-in-the-loop gates:** Updates show a full preview before applying. Deletes require explicit `yes`. When a query matches multiple notes, the agent lists them and asks the user to pick.

---

## Project Structure

```
├── main.py               # CLI entry point
├── src/
│   ├── agent.py          # State machine & flow handlers
│   ├── llm.py            # All LLM calls (intent routing, note rewriting)
│   ├── schemas.py        # Pydantic output schemas (IntentResult, RewrittenNote)
│   ├── store.py          # SQLite CRUD
│   ├── vector_store.py   # Qdrant embeddings
│   ├── models.py         # Note dataclass
│   ├── state.py          # AgentState TypedDict
│   └── config.py         # All env vars loaded once here
└── tests/
    ├── test_store.py
    ├── test_agent.py
    └── test_vector_store.py
```

---

## Setup

**Requirements:** Python 3.13+, `uv`

```bash
uv sync
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemma-4-31b-it
GEMINI_EMBEDDING_MODEL=gemini-embedding-2

QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=notes

MEMORY_SIZE=10
TOP_K_KEYWORD=3
TOP_K_TAG=3
TOP_K_SEMANTIC=1
TOP_K_CANDIDATES=3
```

Run the agent:

```bash
uv run python main.py
```

---

## CLI Examples

### Saving a note

```
You: Save a note about our backend refactoring. We're migrating to FastAPI, keeping PostgreSQL and Redis. Target Q3. Tag: architecture, backend.

[TRACKING] Intent Router -> intent='save'
[TRACKING] Executing SAVE handler -> title='Backend Refactoring Decisions', tags=['architecture', 'backend']
[TRACKING] [SQLite] Inserted note ID: 4f8b9a12
[TRACKING] [Qdrant] Upserted embedding for note ID: 4f8b9a12

Agent: ✅ Saved: 'Backend Refactoring Decisions' (ID: 4f8b9a12) | Tags: architecture, backend
```

---

### Semantic search

```
You: What did we decide about the backend framework?

[TRACKING] Intent Router -> intent='search_semantic'
[TRACKING] [Qdrant] Semantic search -> returned 1 point ID(s)

Agent: Found 1 note(s):

• [4f8b9a12] Backend Refactoring Decisions | Tags: architecture, backend
  We're migrating to FastAPI, keeping PostgreSQL and Redis. Target Q3.
```

---

### Update with preview confirmation

```
You: Update the backend note to add Apache Kafka for event processing.

[TRACKING] Intent Router -> intent='update'
[TRACKING] Searching candidates across Qdrant and SQLite...
[TRACKING] Exactly 1 candidate found: 'Backend Refactoring Decisions' (4f8b9a12)
[TRACKING] LLM Rewrite -> updating note with instruction
[TRACKING] Transitioning to mode: AWAITING_CONFIRM

Agent: Here is how 'Backend Refactoring Decisions' will look after the update:

  Title : Backend Refactoring Decisions
  Body  : Migrating to FastAPI, keeping PostgreSQL, Redis, and adding Apache Kafka for event processing. Target Q3.
  Tags  : architecture, backend

Confirm update? (yes / no)

[AWAITING_CONFIRM] You: yes

[TRACKING] [SQLite] Updated note ID: 4f8b9a12
[TRACKING] [Qdrant] Upserted embedding for note ID: 4f8b9a12

Agent: ✅ Updated 'Backend Refactoring Decisions' (ID: 4f8b9a12).
```

---

### Disambiguation — multiple matches

```
You: Delete the Python note

[TRACKING] Intent Router -> intent='delete'
[TRACKING] Candidates found total: 2
[TRACKING] Multiple candidates -> transitioning to mode: AWAITING_DISAMBIGUATION

Agent: Found 2 notes that could match. Which one did you mean?

  [1] 'Backend Refactoring Decisions' (ID: 4f8b9a12)
  [2] 'Python Asyncio Tutorial' (ID: 9a2c1103)

Reply with the number (e.g. 1) or 'cancel'.

[AWAITING_DISAMBIGUATION] You: 2

[TRACKING] User selected: 'Python Asyncio Tutorial' (9a2c1103)
[TRACKING] Transitioning to mode: AWAITING_CONFIRM

Agent: Are you sure you want to delete 'Python Asyncio Tutorial' (ID: 9a2c1103)? (yes / no)
```

---

### Chitchat

```
You: Hello! What can you do?

[TRACKING] Intent Router -> intent='chitchat'

Agent: Hello! I'm your note-taking assistant. I can save notes, search by keyword, tag, or meaning, and update or delete notes safely. What would you like to do?
```

---
