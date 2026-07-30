"""
Qdrant Vector Store Integration for Semantic Search.

Maintains a lightweight vector index mapping vector embeddings of note titles
and bodies to note_id payloads. If Qdrant or embedding service is offline,
falls back gracefully.
"""

import hashlib
import math
import os
import re
from typing import Optional
from models import Note

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class SimpleEmbeddingProvider:
    """
    Embedding Provider with API embedding support (LangChain Google GenAI) and
    deterministic offline fallback.
    """

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        self._embedder_inst = None

        if self.api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._embedder_inst = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    google_api_key=self.api_key,
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize GoogleGenerativeAIEmbeddings: {e}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for text. Uses LangChain Google GenAI embeddings if configured,
        otherwise generates a normalized term-frequency feature vector.
        """
        if self._embedder_inst:
            try:
                vec = self._embedder_inst.embed_query(text)
                if vec:
                    return vec
            except Exception as e:
                print(f"[Warning] Remote embedding generation failed, using local fallback: {e}")

        return self._local_deterministic_embedding(text)

    def _local_deterministic_embedding(self, text: str) -> list[float]:
        """
        Generate deterministic normalized hash-bag-of-words vector.
        Ensures semantic/keyword similarity works offline in unit tests.
        """
        vector = [0.0] * self.vector_dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vector

        for word in words:
            # Map word hash to vector dimension bucket
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.vector_dim
            vector[idx] += 1.0

        # L2 Normalize
        norm = math.sqrt(sum(val * val for val in vector))
        if norm > 0:
            vector = [val / norm for val in vector]

        return vector



class VectorNoteStore:
    """
    Qdrant Vector Store wrapper for indexing and semantic retrieval.
    Payload strictly contains note_id (canonical text stays in SQLite).
    """

    COLLECTION_NAME = "notes_semantic_index"

    def __init__(
        self,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        vector_dim: int = 128,
    ):
        self.vector_dim = vector_dim
        self.embedder = SimpleEmbeddingProvider(vector_dim=self.vector_dim)

        # Auto-detect actual vector dimension from embedder
        try:
            sample_vector = self.embedder.embed_text("dimension_check")
            if sample_vector and len(sample_vector) > 0:
                self.vector_dim = len(sample_vector)
        except Exception:
            pass

        if not QDRANT_AVAILABLE:
            print("[Warning] qdrant-client not installed. Vector search disabled.")
            self.client = None
            return

        env_url = url or os.environ.get("QDRANT_URL")
        env_api_key = api_key or os.environ.get("QDRANT_API_KEY")

        if env_url and env_api_key:
            self.client = QdrantClient(url=env_url, api_key=env_api_key)
        else:
            loc = location if location is not None else ":memory:"
            self.client = QdrantClient(location=loc)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client:
            return

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME in collections:
                # Inspect existing collection vector dimension
                collection_info = self.client.get_collection(self.COLLECTION_NAME)
                current_size = None
                if hasattr(collection_info.config.params, "vectors"):
                    v_config = collection_info.config.params.vectors
                    if hasattr(v_config, "size"):
                        current_size = v_config.size

                if current_size and current_size != self.vector_dim:
                    print(
                        f"[Info] Recreating Qdrant collection '{self.COLLECTION_NAME}' "
                        f"(dimension change: {current_size} -> {self.vector_dim})"
                    )
                    self.client.delete_collection(self.COLLECTION_NAME)
                    collections.remove(self.COLLECTION_NAME)

            if self.COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            print(f"[Warning] Failed to initialize Qdrant collection: {e}")


    def upsert_note(self, note: Note) -> bool:
        """
        Embed note title and body, then upsert into Qdrant index.
        Payload stores only note_id.
        """
        if not self.client:
            return False

        try:
            text_to_embed = f"{note.title}\n{note.body}"
            vector = self.embedder.embed_text(text_to_embed)

            # Use deterministic integer ID derived from UUID string for Qdrant PointStruct
            point_id = int(hashlib.md5(note.id.encode("utf-8")).hexdigest()[:16], 16)

            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"note_id": note.id},
                    )
                ],
            )
            return True
        except Exception as e:
            print(f"[Warning] Failed to upsert vector for note {note.id}: {e}")
            return False

    def delete_note(self, note_id: str) -> bool:
        """
        Delete vector embedding for note_id from Qdrant index.
        """
        if not self.client:
            return False

        try:
            point_id = int(hashlib.md5(note_id.encode("utf-8")).hexdigest()[:16], 16)
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[point_id],
            )
            return True
        except Exception as e:
            print(f"[Warning] Failed to delete vector for note {note_id}: {e}")
            return False

    def search(self, query: str, top_k: int = 5, score_threshold: Optional[float] = None) -> list[str]:
        """
        Search vector index by natural language query.
        Returns list of matching note_ids ranked by similarity score,
        filtering out low-relevance matches below score_threshold.
        """
        if not self.client or not query.strip():
            return []

        if score_threshold is None:
            # Gemini API dense embeddings use 0.30 threshold; offline sparse fallback uses 0.10 threshold
            score_threshold = 0.30 if self.embedder.api_key else 0.10

        try:
            query_vector = self.embedder.embed_text(query)
            
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                search_results = response.points
            elif hasattr(self.client, "search"):
                search_results = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
            else:
                return []

            return [
                res.payload["note_id"]
                for res in search_results
                if res.payload and "note_id" in res.payload and getattr(res, "score", 1.0) >= score_threshold
            ]
        except Exception as e:
            print(f"[Warning] Qdrant vector search failed, falling back: {e}")
            return []



