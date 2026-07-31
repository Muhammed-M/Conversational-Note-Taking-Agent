"""
config.py — Single source of truth for all app settings.

Reads every setting from the .env file and exposes them as constants.
Every other file imports from here. No other file calls os.getenv() directly.
"""

import os
from dotenv import load_dotenv

# Load the .env file so os.getenv() can read it
load_dotenv()


# ── Gemini LLM ────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")               # model used for intent parsing and note rewriting
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")  # model used to embed notes into vectors


# ── Qdrant Vector Database ────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL")                                       # your Qdrant cloud cluster URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")                               # your Qdrant cloud API key
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")                # name of the Qdrant collection to use


# ── Agent Behavior ────────────────────────────────────────────────────────────

MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", "10"))
# how many past messages from the conversation are sent to the LLM as context

TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "3"))
# keyword search: how many notes to return when searching by a specific word

TOP_K_TAG = int(os.getenv("TOP_K_TAG", "3"))
# tag search: how many notes to return when searching by tags

TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "1"))
# semantic search: how many notes to return from Qdrant vector search

TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "3"))
# update/delete: how many candidate notes to show the user when multiple notes match
