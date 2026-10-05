import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .document_loader import Document

@dataclass
class TextChunk:
    """Represents a chunked segment of an enterprise document."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

class RecursiveTokenChunker:
    """
    Splits text recursively using hierarchical separators:
    ['\n\n# ', '\n\n## ', '\n\n### ', '\n\n', '\n', '. ', ' ', '']
    with configurable target chunk size (in tokens/characters) and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n# ",
            "\n\n## ",
            "\n\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def _approx_token_count(self, text: str) -> int:
        """Approximate token count (rule-of-thumb: 1 token ~ 4 chars in English)."""
        return max(1, len(text) // 4)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using the given hierarchy of separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if separator and separator != " " and separator != "":
                piece = s if s.startswith(separator) else separator + s
            else:
                piece = s

            if self._approx_token_count(piece) <= self.chunk_size:
                good_splits.append(piece.strip())
            else:
                if new_separators:
                    sub_splits = self._split_text(piece, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard cut if no more separators
                    for j in range(0, len(piece), self.chunk_size * 4):
                        good_splits.append(piece[j: j + self.chunk_size * 4].strip())

        # Merge small splits respecting overlap
        merged: List[str] = []
        curr: List[str] = []
        curr_len = 0

        for piece in good_splits:
            piece_len = self._approx_token_count(piece)
            if curr_len + piece_len > self.chunk_size:
                if curr:
                    merged.append(" ".join(curr))
                    # Retain overlap from end of current buffer
                    overlap_buffer: List[str] = []
                    overlap_len = 0
                    for rev_p in reversed(curr):
                        p_len = self._approx_token_count(rev_p)
                        if overlap_len + p_len <= self.chunk_overlap:
                            overlap_buffer.insert(0, rev_p)
                            overlap_len += p_len
                        else:
                            break
                    curr = overlap_buffer
                    curr_len = overlap_len
            curr.append(piece)
            curr_len += piece_len

        if curr:
            merged.append(" ".join(curr))

        return merged

    def split_document(self, document: Document) -> List[TextChunk]:
        """Split a document into TextChunk objects with deterministic IDs and metadata."""
        raw_chunks = self._split_text(document.content, self.separators)
        chunks: List[TextChunk] = []

        for idx, text in enumerate(raw_chunks):
            cleaned = text.strip()
            if not cleaned:
                continue

            chunk_hasher = hashlib.sha256()
            chunk_hasher.update(f"{document.doc_id}_{idx}_{cleaned[:50]}".encode("utf-8"))
            chunk_id = chunk_hasher.hexdigest()[:16]

            meta = dict(document.metadata)
            meta.update({
                "chunk_index": idx,
                "total_estimated_chunks": len(raw_chunks),
                "parent_doc_id": document.doc_id,
            })

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=cleaned,
                    chunk_index=idx,
                    metadata=meta,
                    token_count=self._approx_token_count(cleaned),
                )
            )

        return chunks

    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Split multiple documents into a flat list of TextChunks."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
