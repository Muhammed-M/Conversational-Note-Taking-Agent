"""
vector_store.py — Qdrant vector store for semantic search.
"""

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config
from src.models import Note


class VectorStore:
    """Qdrant vector store wrapper for note embeddings."""

    def __init__(self):
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

        self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.collection = config.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist yet."""
        existing_names = [c.name for c in self.client.get_collections().collections]

        if self.collection not in existing_names:
            sample_vector = self.embedder.embed_query("hello")
            dimension = len(sample_vector)

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )

    def _note_id_to_point_id(self, note_id: str) -> int:
        """Convert note UUID string to integer using MD5 hashing."""
        return int(hashlib.md5(note_id.encode()).hexdigest()[:16], 16)

    def upsert_note(self, note: Note) -> None:
        """Embed note title and body, then store vector in Qdrant."""
        text = f"{note.title}\n{note.body}"
        vector = self.embedder.embed_query(text)

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._note_id_to_point_id(note.id),
                    vector=vector,
                    payload={"note_id": note.id},
                )
            ],
        )

    def delete_note(self, note_id: str) -> None:
        """Remove a note's embedding from Qdrant by note ID."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=[self._note_id_to_point_id(note_id)],
        )

    def search(self, query: str, top_k: int = None) -> list[str]:
        """Search Qdrant for notes semantically similar to the query."""
        top_k = top_k or config.TOP_K_SEMANTIC

        query_vector = self.embedder.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )

        return [
            point.payload["note_id"]
            for point in response.points
            if point.payload and "note_id" in point.payload
        ]
