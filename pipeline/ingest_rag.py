#!/usr/bin/env python3
"""
RAG Ingestion Pipeline for RBI Policy Documents
================================================

Reads all PDFs from rules/rbi_guidelines/, splits into chunks,
embeds with sentence-transformers, and stores in ChromaDB.

Cloud routing (mirrors agents/rag_agent.py):
  CHROMA_CLOUD_API_KEY + CHROMA_CLOUD_TENANT set  →  Chroma Cloud
  otherwise                                        →  local PersistentClient

Run once (or whenever you add new PDFs):
    python pipeline/ingest_rag.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# ── project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

# ── lazy-install guards ───────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    os.system("pip install PyMuPDF -q")
    import fitz

try:
    import chromadb
except ImportError:
    os.system("pip install chromadb -q")
    import chromadb

# Stub gRPC telemetry exporter before chromadb loads it — prevents DLL
# failures on Windows machines where cygrpc.pyd is blocked by AppControl.
import sys as _sys, types as _types

class _NullStub:
    def __getattr__(self, name): return _NullStub()
    def __call__(self, *a, **kw): return _NullStub()

for _grpc_mod in [
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
]:
    if _grpc_mod not in _sys.modules:
        _stub = _types.ModuleType(_grpc_mod)
        _stub.OTLPSpanExporter   = _NullStub
        _stub.OTLPLogExporter    = _NullStub
        _stub.OTLPMetricExporter = _NullStub
        _sys.modules[_grpc_mod]  = _stub

# RecursiveCharacterTextSplitter — pure Python implementation.
# We avoid importing langchain_text_splitters at the package level because its
# __init__.py unconditionally imports sentence_transformers → torch, which
# fails on Windows machines with AppControl policies blocking torch DLLs.
# This inline implementation is functionally identical to LangChain's version.

import re as _re
from typing import Iterable as _Iterable

class RecursiveCharacterTextSplitter:
    """
    Pure-Python recursive character splitter.
    Mirrors LangChain's RecursiveCharacterTextSplitter interface.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list = None,
        length_function=len,
        **kwargs,
    ):
        self._chunk_size     = chunk_size
        self._chunk_overlap  = chunk_overlap
        self._separators     = separators or ["\n\n", "\n", ". ", " ", ""]
        self._length_function = length_function

    def _split_text(self, text: str, separators: list) -> list:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if _re.search(sep, text):
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = _re.split(separator, text) if separator else list(text)
        _good, _current, _current_len = [], [], 0

        for s in splits:
            s_len = self._length_function(s)
            if _current_len + s_len + (1 if _current else 0) > self._chunk_size:
                if _current:
                    merged = separator.join(_current).strip()
                    if merged:
                        final_chunks.append(merged)
                    # keep overlap
                    while _current and _current_len > self._chunk_overlap:
                        _current_len -= self._length_function(_current[0]) + len(separator)
                        _current.pop(0)
            _current.append(s)
            _current_len += s_len + (len(separator) if len(_current) > 1 else 0)

        if _current:
            merged = separator.join(_current).strip()
            if merged:
                final_chunks.append(merged)

        result = []
        for chunk in final_chunks:
            if self._length_function(chunk) > self._chunk_size and new_separators:
                result.extend(self._split_text(chunk, new_separators))
            else:
                result.append(chunk)
        return result

    def split_text(self, text: str) -> list:
        return self._split_text(text, self._separators)

    def split_documents(self, documents) -> list:
        """Split a list of Document objects."""
        chunks = []
        for doc in documents:
            for chunk_text in self.split_text(doc.page_content):
                if chunk_text.strip():
                    # Create a new Document-like object
                    new_doc = type(doc)(
                        page_content=chunk_text,
                        metadata=dict(doc.metadata),
                    )
                    chunks.append(new_doc)
        return chunks

try:
    from langchain_core.documents import Document
except ImportError:
    # Simple dataclass fallback — no external deps
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        page_content: str
        metadata: dict = field(default_factory=dict)

# sentence_transformers loads torch at import time which is blocked by Windows
# AppControl policies.  We use chromadb's built-in DefaultEmbeddingFunction
# instead — it bundles onnxruntime (no torch DLL) and is fully compatible
# with Chroma Cloud collections created via this pipeline.
try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEmbedFn
    _USE_CHROMA_EMBED = True
except Exception:
    _USE_CHROMA_EMBED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB factory — matches agents/rag_agent.py routing
# ─────────────────────────────────────────────────────────────────────────────

def _build_chroma_client():
    """Return (client, mode) — Cloud if keys present, else local PersistentClient."""
    api_key  = os.getenv("CHROMA_CLOUD_API_KEY", "")
    tenant   = os.getenv("CHROMA_CLOUD_TENANT",  "")
    database = os.getenv("CHROMA_CLOUD_DATABASE", "rbi_guidelines")

    if api_key and tenant:
        try:
            client = chromadb.CloudClient(
                tenant   = tenant,
                database = database,
                api_key  = api_key,
            )
            client.heartbeat()
            logger.info("☁️  Chroma Cloud connected  (tenant=%s  db=%s)", tenant, database)
            return client, "cloud"
        except Exception as exc:
            logger.warning("⚠️  Chroma Cloud unavailable (%s) — falling back to local", exc)

    local_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/embeddings")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=local_dir)
    logger.info("💾 Local Chroma at %s", local_dir)
    return client, "local"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class RAGIngestionPipeline:

    def __init__(self):
        root = Path(__file__).parent.parent
        self.pdf_directory   = root / "rules" / "rbi_guidelines"
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rbi_guidelines")
        self.chunk_size      = 1000
        self.chunk_overlap   = 200
        self._embed_fn       = None

        logger.info("PDF source : %s", self.pdf_directory)

    # ── embedding function ────────────────────────────────────────────────

    def _get_embed_fn(self):
        """
        Returns a chromadb-compatible embedding function.
        Uses DefaultEmbeddingFunction (onnxruntime-based, no torch required).
        Falls back to a no-op that lets Chroma embed server-side if needed.
        """
        if self._embed_fn is None:
            if _USE_CHROMA_EMBED:
                logger.info("Using chromadb DefaultEmbeddingFunction (ONNX, no torch)")
                self._embed_fn = _ChromaEmbedFn()
            else:
                logger.warning("No embedding function available — Chroma will use its default")
                self._embed_fn = None
        return self._embed_fn

    # ── PDF loading ───────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        try:
            doc  = fitz.open(str(path))
            text = "".join(page.get_text() for page in doc)
            doc.close()
            text = text.replace("\n\n", "\n").strip()
            logger.info("  📄 %-55s  %d chars", path.name[:55], len(text))
            return text
        except Exception as exc:
            logger.error("  ✗ %s : %s", path.name, exc)
            return ""

    def load_documents(self) -> List[Document]:
        pdfs = sorted(self.pdf_directory.glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found in %s", self.pdf_directory)
            return []
        logger.info("Found %d PDF files", len(pdfs))
        docs = []
        for p in pdfs:
            text = self._extract_pdf(p)
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":        str(p),
                        "filename":      p.name,
                        "document_type": "rbi_guideline",
                        "ingested_at":   datetime.now().isoformat(),
                        "char_count":    len(text),
                    },
                ))
        logger.info("✅ Loaded %d documents", len(docs))
        return docs

    # ── chunking ──────────────────────────────────────────────────────────

    def split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = ["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata.update({"chunk_id": i, "chunk_size": len(c.page_content)})
        logger.info("✅ Split into %d chunks", len(chunks))
        return chunks

    # ── ChromaDB ingestion ────────────────────────────────────────────────

    def ingest(self, chunks: List[Document], client, mode: str):
        """Delete-then-recreate collection, then batch-insert all chunks."""
        try:
            client.delete_collection(self.collection_name)
            logger.info("🗑️  Deleted old collection '%s'", self.collection_name)
        except Exception:
            pass  # didn't exist yet

        embed_fn = self._get_embed_fn()

        collection = client.create_collection(
            name               = self.collection_name,
            embedding_function = embed_fn,
            metadata           = {"hnsw:space": "cosine",
                                  "description": "RBI Guidelines — Loan Decision System"},
        )
        logger.info("📦 Collection '%s' created (%s)", self.collection_name, mode)

        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata     for c in chunks]
        ids       = [f"chunk_{i}"   for i in range(len(chunks))]
        batch_size = 50   # smaller batches for cloud API stability

        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            collection.add(
                documents = texts[start:end],
                metadatas = metadatas[start:end],
                ids       = ids[start:end],
            )
            batch_no = start // batch_size + 1
            total    = (len(texts) + batch_size - 1) // batch_size
            logger.info("  batch %d/%d  (+%d docs)", batch_no, total, len(texts[start:end]))

        final_count = collection.count()
        logger.info("✅ Stored %d chunks in ChromaDB (%s)", final_count, mode)
        return final_count

    # ── verification ──────────────────────────────────────────────────────

    def verify(self, client):
        col   = client.get_collection(self.collection_name)
        count = col.count()
        result = col.query(query_texts=["loan application FOIR limit"], n_results=2)
        sample = result["documents"][0][0][:120] if result["documents"][0] else "(empty)"
        logger.info("🔍 Test query → '%s ...'", sample)
        return count > 0

    # ── main run ──────────────────────────────────────────────────────────

    def run(self) -> bool:
        print("\n" + "═" * 60)
        print("  RAG INGESTION PIPELINE")
        print("═" * 60)

        client, mode = _build_chroma_client()

        docs = self.load_documents()
        if not docs:
            logger.error("No documents to ingest. Add PDFs to %s", self.pdf_directory)
            return False

        chunks = self.split_documents(docs)
        count  = self.ingest(chunks, client, mode)
        ok     = self.verify(client)

        print("\n" + "═" * 60)
        if ok:
            print(f"  ✅ INGESTION COMPLETE")
            print(f"     Backend  : {mode.upper()}")
            print(f"     PDFs     : {len(docs)}")
            print(f"     Chunks   : {count}")
            print(f"     Collection: {self.collection_name}")
        else:
            print("  ❌ INGESTION FAILED — check logs above")
        print("═" * 60 + "\n")
        return ok


# ─────────────────────────────────────────────────────────────────────────────

def main():
    pipeline = RAGIngestionPipeline()
    success  = pipeline.run()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
