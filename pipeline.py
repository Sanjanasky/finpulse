"""
FinPulse RAG — Pipeline Orchestrator (Local, Free)

Wires all stages: load → chunk → embed → store → retrieve → generate

Usage:
    from pipeline import FinPulsePipeline

    pipeline = FinPulsePipeline(ollama_model="llama3.2")
    pipeline.ingest_directory("data/sample_docs")
    answer = pipeline.ask("What was Apple's Q3 2024 revenue?")
    print(answer.display())
"""

from pathlib import Path
from typing import Optional

from ingestion.embedder import get_embedder, DEFAULT_MODEL
from ingestion.loader import FinancialDocumentLoader, RecursiveChunker
from generation.generator import FinPulseGenerator, get_generator
from retrieval.retriever import FinPulseRetriever
from retrieval.vector_store import FAISSVectorStore


class FinPulsePipeline:
    """
    End-to-end local RAG pipeline — no API keys, no cost.

    Dependencies:
      - sentence-transformers  (embeddings, auto-downloaded ~22MB)
      - faiss-cpu              (vector search)
      - Ollama                 (LLM, install separately)

    Stages:
      1. Load documents (.txt / .md / .pdf)
      2. Chunk with overlap
      3. Embed with all-MiniLM-L6-v2 (local)
      4. Store in FAISS index
      5. On query: embed → MMR search → prompt → Ollama → parse
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        retrieval_strategy: str = "mmr",
        embedding_model: str = DEFAULT_MODEL,
        ollama_model: str = "llama3.2",
        force_mock: bool = False,
    ):
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy

        print("\n🔧 Initialising FinPulse RAG Pipeline (local mode)...")

        self.loader = FinancialDocumentLoader()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = get_embedder(model_name=embedding_model)
        self.store = FAISSVectorStore(dim=self.embedder.dim)
        self.retriever = FinPulseRetriever(self.store, self.embedder)
        self.generator = FinPulseGenerator(
            llm=get_generator(model=ollama_model, force_mock=force_mock),
            model=ollama_model,
        )

        print("✅ Pipeline ready.\n")

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> int:
        print(f"  📄 Loading: {Path(file_path).name}")
        doc = self.loader.load_file(file_path)
        return self._index_documents([doc])

    def ingest_directory(self, dir_path: str) -> int:
        print(f"  📁 Scanning: {dir_path}")
        docs = self.loader.load_directory(dir_path)
        if not docs:
            print("  ⚠  No documents found. Add .txt/.md/.pdf files.")
            return 0
        print(f"  ✓ Loaded {len(docs)} document(s)")
        return self._index_documents(docs)

    def ingest_text(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> int:
        doc = self.loader.load_text(text, doc_id=doc_id, metadata=metadata or {})
        return self._index_documents([doc])

    def _index_documents(self, docs) -> int:
        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            return 0
        print(f"  ✓ Created {len(chunks)} chunk(s)")
        print("  🔢 Embedding locally...")
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add_chunks(chunks, embeddings)
        print(f"  ✓ Indexed {len(chunks)} chunks — store total: {self.store.size}")
        return len(chunks)

    # ── Query ─────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: Optional[int] = None, verbose: bool = False):
        if self.store.size == 0:
            raise RuntimeError("No documents indexed. Call ingest_*() first.")

        k = top_k or self.top_k
        print(f"\n🔍 Retrieving top-{k} chunks...")
        context = self.retriever.retrieve(question, top_k=k, strategy=self.retrieval_strategy)

        if verbose:
            print("\n--- Retrieved Context ---")
            for r in context.results:
                print(f"  [{r.rank}] score={r.score:.3f} | {r.chunk.chunk_id}")
                print(f"       {r.chunk.text[:120]}...")
            print("---\n")

        print("💬 Generating answer via Ollama...")
        return self.generator.answer(question, context)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str = "./finpulse_index") -> None:
        self.store.save(directory)

    def load(self, directory: str = "./finpulse_index") -> None:
        self.store.load(directory)
