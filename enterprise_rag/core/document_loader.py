import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Document:
    """Representation of an ingested enterprise document."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic ID based on source and content hash
            src = self.metadata.get("source", "inline")
            hasher = hashlib.sha256()
            hasher.update((src + ":" + self.content[:200]).encode("utf-8"))
            self.doc_id = hasher.hexdigest()[:16]

class DocumentLoader:
    """Loads enterprise documents from various formats (Text, Markdown, PDF, Directory)."""

    @staticmethod
    def load_text(file_path: str) -> Document:
        """Load standard text or markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return Document(
            content=content,
            metadata={
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "file_size": os.path.getsize(file_path),
            }
        )

    @staticmethod
    def load_pdf(file_path: str) -> Document:
        """Extract text from PDF using pypdf or PyMuPDF, with graceful text extraction fallback."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
            
        content = ""
        # Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(pages_text)
        except ImportError:
            # Try fitz / pymupdf
            try:
                import fitz
                doc = fitz.open(file_path)
                pages_text = [page.get_text() for page in doc]
                content = "\n\n".join(pages_text)
            except ImportError:
                # Fallback: extract binary ASCII stream if no library available
                with open(file_path, "rb") as f:
                    raw = f.read()
                # Basic string extraction for uncompressed streams
                content = f"[Extracted from PDF binary: {os.path.basename(file_path)}]\n" + "".join(
                    chr(b) for b in raw if 32 <= b <= 126 or b in (10, 13)
                )[:2000]

        return Document(
            content=content.strip(),
            metadata={
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "extension": ".pdf",
                "file_size": os.path.getsize(file_path),
            }
        )

    @classmethod
    def load_file(cls, file_path: str) -> Document:
        """Auto-detect format and load document."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path)
        return cls.load_text(file_path)

    @classmethod
    def load_directory(cls, dir_path: str, extensions: Optional[List[str]] = None) -> List[Document]:
        """Load all supported documents in a directory recursively."""
        if extensions is None:
            extensions = [".txt", ".md", ".pdf", ".json", ".csv"]
        
        docs = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, file)
                    try:
                        docs.append(cls.load_file(full_path))
                    except Exception as e:
                        print(f"Warning: Failed to load {full_path}: {e}")
        return docs
