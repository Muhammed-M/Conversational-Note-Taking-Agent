# Conversational Note-Taking Agent

A production-grade conversational note-taking CLI agent built for the TechLabs London AI Engineer Assessment. 

The system enables users to manage personal notes entirely through natural language conversation, featuring **Intent Disambiguation**, **Confirmation Gates for Destructive Actions**, **Multi-Turn Context Awareness**, and **Hybrid Vector-SQLite Retrieval**.

---

## 🏗️ Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │      Natural Language User      │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │    NoteAgentGraph (LangGraph)   │
                        └───────┬─────────────────┬───────┘
                                │                 │
            ┌───────────────────┘                 └───────────────────┐
            ▼                                                         ▼
┌───────────────────────┐                                 ┌───────────────────────┐
│     SQLite Store      │ ◄────── Sync Observers ────────►│  Qdrant Vector Store  │
│  (Canonical Database) │  (on_created/updated/deleted)   │   (Semantic Index)    │
└───────────────────────┘                                 └───────────────────────┘
```

### 1. Dual-Storage Design & Sync Rule
- **SQLite (`store.py`)**: Canonical source of truth. All CRUD operations mutate SQLite first. SQLite also handles structured filters (tag exact matches, ISO date ranges).
- **Qdrant Vector Index (`vector_store.py`)**: Vector similarity index where payloads contain **only `note_id`** (preventing payload duplication and data drift).
- **Sync Rule**: SQLite mutation events register observers (`on_created`, `on_updated`, `on_deleted`). Whenever a note is inserted, updated, or removed in SQLite, the vector index automatically syncs.
- **Resilience**: If Qdrant or embedding APIs are offline, search gracefully degrades to SQLite `LIKE` keyword search without crashing.

### 2. Safety & Control Flow Gates
- **Intent Disambiguation Gate (`AWAITING_DISAMBIGUATION`)**: When a user's request matches 2+ candidate notes (e.g., "Delete the project review note"), execution pauses and prompts the user to pick a specific candidate by number before executing.
- **Confirmation Gate (`AWAITING_CONFIRM`)**: Destructive actions (`delete` or significant `update`) pause the graph, displaying the exact target note title and ID, requiring explicit user confirmation (`yes` / `no`) before writing to the database.
- **Multi-Turn Anaphora Resolution**: Tracks `last_note_id` in state. If a user follows up with *"Actually, add a deadline to that last note"* or *"Delete it"*, the agent seamlessly resolves the target note.

---

## 🛠️ Project Structure

```
.
├── models.py              # Domain model Note entity & UTC timestamp helpers
├── store.py               # SQLite canonical store with observer callbacks
├── vector_store.py        # Qdrant client wrapper & embedding fallback
├── tools.py               # Typed Pydantic tool schemas for LLM function calls
├── state.py               # AgentState TypedDict definition
├── graph.py               # LangGraph state machine execution & safety gates
├── main.py                # Interactive CLI entry point
├── pyproject.toml         # Package definition & dependencies
├── .env.example           # Environment template (GEMINI_API_KEY)
└── tests/                 # Automated test suite
    ├── test_store.py            # Unit tests for SQLite CRUD & observers
    ├── test_vector_store.py     # Unit tests for Qdrant index & search
    ├── test_disambiguation.py   # Unit tests for Intent Disambiguation gate
    └── test_confirmation_flow.py# Unit tests for Confirmation gate & multi-turn
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies using `pip` or `uv`:

```bash
# Using pip
pip install -e .

# Or using uv
uv sync
```

### 2. Set API Key (Optional for Remote Embeddings & Gemini)
Copy `.env.example` to `.env` and add your Google Gemini API key:

```bash
cp .env.example .env
```

> **Note**: If `GEMINI_API_KEY` is omitted, the agent automatically falls back to an offline deterministic feature-vector embedder and rule parser, allowing 100% offline functionality.

### 3. Run Interactive CLI
```bash
python main.py
```

### 4. Run Test Suite
```bash
python -m pytest tests/
```

---

## 💡 Example Conversational Flows

### 1. Add Note
```
User: Save a note about team standup — we agreed to move it to Tuesdays, tag it as meetings.
Agent: Saved note 'Team standup' (ID: a1b2c3d4) with tags: meetings.
```

### 2. Hybrid Search
```
User: What did I write about standup?
Agent: Found 1 note(s):
• 'Team standup' (ID: a1b2c3d4) [tags: meetings]
  we agreed to move it to Tuesdays
```

### 3. Intent Disambiguation
```
User: Delete the standup note
(If multiple standup notes exist)
Agent: Multiple notes matched your request to delete. Which one did you mean?
[1] 'Team standup' (ID: a1b2c3d4): we agreed to move it to Tuesdays...
[2] 'Standup agenda' (ID: e5f6g7h8): review sprint goals...
Reply with the number (e.g. 1) or 'cancel'.

User: 1
Agent: Are you sure you want to DELETE note 'Team standup' (ID: a1b2c3d4)? (yes / no)
```

### 4. Multi-Turn Anaphora ("That note")
```
User: Add a note: Buy groceries : milk, eggs, bread
Agent: Saved note 'Buy groceries' (ID: 9x8y7z6w).

User: Actually add apples to that last note
Agent: Are you sure you want to UPDATE note 'Buy groceries' (ID: 9x8y7z6w)? (yes / no)
```

---

## 🎓 Technical Interview Guide (How to Explain Code Decisions)

If asked during the technical review:

1. **Why SQLite + Qdrant?**
   > *"SQLite is our canonical relational store ensuring ACID guarantees and exact SQL filtering. Qdrant serves exclusively as a vector index storing `note_id` payloads, which eliminates data duplication and payload sync bugs."*

2. **How is storage synchronization guaranteed?**
   > *"We implemented an Observer / Callback pattern inside `NoteStore`. Any call to `add_note`, `update_note`, or `delete_note` automatically fires registered listeners, ensuring Qdrant is updated atomically from the canonical store."*

3. **How do Safety Gates work?**
   > *"Rather than letting the LLM directly invoke destructive side-effects, tool intent flows through state machine nodes (`AWAITING_DISAMBIGUATION` and `AWAITING_CONFIRM`). If candidates > 1, it forces disambiguation. For updates and deletes, it explicitly prompts for user confirmation (`yes`/`no`) before committing to disk."*
