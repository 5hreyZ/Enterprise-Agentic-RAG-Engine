import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enterprise_rag.core.chunking import TextChunk
from enterprise_rag.core.embeddings import EmbeddingEngine
from enterprise_rag.config import settings

@dataclass
class SearchResult:
    """Standardized retrieval search result across dense, sparse, and reranked stages."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "dense"  # "dense", "sparse", "hybrid", "reranked"

class DenseRetriever:
    """
    Production Dense Retriever backed by Qdrant Vector DB with in-memory fallback.
    Encodes queries and documents using dense embeddings and performs Cosine/Dot-Product search.
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        collection_name: Optional[str] = None,
    ):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._qdrant_client = None
        self._in_memory_store: List[Dict[str, Any]] = []
        self._init_qdrant()

    def _init_qdrant(self):
        """Initialize Qdrant client connection if available, otherwise use in-memory store."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                timeout=5.0,
            )
            # Check or create collection
            collections = [c.name for c in client.get_collections().collections]
            if self.collection_name not in collections:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_engine.dimension,
                        distance=Distance.COSINE,
                    ),
                )
            self._qdrant_client = client
        except (ImportError, Exception):
            # Fallback to high-performance in-memory vector index
            self._qdrant_client = None

    def index_chunks(self, chunks: List[TextChunk]) -> int:
        """Embed and index chunks into Qdrant or in-memory vector store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embedding_engine.embed_batch(texts)

        if self._qdrant_client is not None:
            from qdrant_client.http.models import PointStruct
            points = [
                PointStruct(
                    id=c.chunk_id,
                    vector=vec,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    },
                )
                for c, vec in zip(chunks, vectors)
            ]
            self._qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return len(points)
        else:
            # In-memory indexing
            for c, vec in zip(chunks, vectors):
                self._in_memory_store.append({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                    "vector": vec,
                })
            return len(chunks)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return dot / (n1 * n2)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform dense semantic similarity search for a query string."""
        query_vec = self.embedding_engine.embed_text(query)

        if self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=top_k,
                )
                results = []
                for hit in hits:
                    results.append(
                        SearchResult(
                            chunk_id=hit.payload["chunk_id"],
                            doc_id=hit.payload["doc_id"],
                            text=hit.payload["text"],
                            score=float(hit.score),
                            metadata=hit.payload.get("metadata", {}),
                            source="dense",
                        )
                    )
                return results
            except Exception as e:
                # On error, fallback to in-memory store
                pass

        # In-memory cosine search
        scored = []
        for item in self._in_memory_store:
            score = self._cosine_similarity(query_vec, item["vector"])
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored[:top_k]

        return [
            SearchResult(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                score=float(score),
                metadata=item["metadata"],
                source="dense",
            )
            for score, item in top_hits
        ]
