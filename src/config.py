"""
config.py — Central configuration module.

All environment variables, API keys, model names, database paths,
and configurable constants are loaded and validated here once.

Other modules MUST import settings from config.py rather than calling os.getenv().
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the .env file in the project root
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _get_required_env(var_name: str) -> str:
    """Read an environment variable. Raise a clear error if it is missing."""
    val = os.getenv(var_name)
    if not val:
        raise ValueError(
            f"Missing required environment variable '{var_name}'. "
            f"Please check your .env file."
        )
    return val


# ── Gemini API Credentials & Model Config ─────────────────────────────────────
GEMINI_API_KEY: str = _get_required_env("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

# ── Qdrant Vector DB Configuration ────────────────────────────────────────────
QDRANT_URL: str = _get_required_env("QDRANT_URL")
QDRANT_API_KEY: str = _get_required_env("QDRANT_API_KEY")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "notes")

# ── SQLite Database Configuration ─────────────────────────────────────────────
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "notes.db"))

# ── Agent Behavior & Limits ───────────────────────────────────────────────────
MEMORY_SIZE: int = int(os.getenv("MEMORY_SIZE", "10"))
TOP_K_KEYWORD: int = int(os.getenv("TOP_K_KEYWORD", "3"))
TOP_K_TAG: int = int(os.getenv("TOP_K_TAG", "3"))
TOP_K_SEMANTIC: int = int(os.getenv("TOP_K_SEMANTIC", "1"))
TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", "3"))
