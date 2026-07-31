"""
vector_store.py — Qdrant vector store for semantic search.

This module handles storing and searching note embeddings in Qdrant cloud.

The flow is:
  1. When a note is saved or updated:
     - Combine title + body into one text string
     - Send it to Gemini to get a vector (list of numbers that represents the meaning)
     - Store that vector in Qdrant, with the note's ID as the payload

  2. When the user searches semantically:
     - Embed the search query with Gemini
     - Search Qdrant for the most similar vectors
     - Return the matching note IDs
     - The agent then fetches the full notes from SQLite using those IDs

We store ONLY the note ID in Qdrant — all actual note data stays in SQLite.
"""

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config
from models import Note


class VectorStore:
    """Qdrant vector store wrapper for note embeddings."""

    def __init__(self):
        # Set up the Gemini embedding model
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

        # Connect to the Qdrant cloud cluster
        self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        # Name of the collection inside Qdrant where we store note embeddings
        self.collection = config.QDRANT_COLLECTION

        # Make sure the collection exists (create it if this is the first run)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't exist yet.
        On the first run, we embed a short test string to find out
        the vector dimension (number of elements) returned by the embedding model.
        """
        existing_names = [c.name for c in self.client.get_collections().collections]

        if self.collection not in existing_names:
            # Find the vector dimension by embedding a sample text
            sample_vector = self.embedder.embed_query("hello")
            dimension = len(sample_vector)

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dimension,           # number of dimensions in each vector
                    distance=Distance.COSINE, # cosine similarity: measures angle between vectors
                ),
            )

    def _note_id_to_point_id(self, note_id: str) -> int:
        """
        Qdrant requires integer IDs for points.
        Convert the note's UUID string to a large integer using MD5 hashing.
        This is deterministic — same note_id always gives same integer.
        """
        return int(hashlib.md5(note_id.encode()).hexdigest()[:16], 16)

    def upsert_note(self, note: Note) -> None:
        """
        Embed the note's title and body, then store (or update) the vector in Qdrant.
        The payload stores only the note_id — full note data stays in SQLite.

        "Upsert" means: insert if new, update if the note already exists.
        """
        # Combine title and body into one string to embed
        text = f"{note.title}\n{note.body}"
        vector = self.embedder.embed_query(text)

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._note_id_to_point_id(note.id),  # Qdrant point ID (integer)
                    vector=vector,                            # the embedding vector
                    payload={"note_id": note.id},            # we store the UUID here so we can look it up later
                )
            ],
        )

    def delete_note(self, note_id: str) -> None:
        """Remove a note's embedding from Qdrant by its note ID."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=[self._note_id_to_point_id(note_id)],
        )

    def search(self, query: str, top_k: int = None) -> list[str]:
        """
        Search Qdrant for notes semantically similar to the query.

        Steps:
          1. Embed the query text into a vector
          2. Ask Qdrant for the top_k most similar vectors
          3. Return the note IDs from the matching payloads

        The agent then uses these IDs to fetch full notes from SQLite.
        """
        top_k = top_k or config.TOP_K_SEMANTIC

        # Embed the search query
        query_vector = self.embedder.embed_query(query)

        # Search for the nearest vectors in Qdrant
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )

        # Extract and return the note_id from each result's payload
        return [
            point.payload["note_id"]
            for point in response.points
            if point.payload and "note_id" in point.payload
        ]
