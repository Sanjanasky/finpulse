"""
FinPulse RAG — Unit Tests
Run with: python -m pytest tests.py -v
All tests run offline — no Ollama or API keys needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest


# ── Loader ──────────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def setup_method(self):
        from ingestion.loader import FinancialDocumentLoader
        self.loader = FinancialDocumentLoader()

    def test_load_text(self):
        doc = self.loader.load_text("Apple Q3 revenue was $85.8B.", doc_id="test_doc")
        assert doc.doc_id == "test_doc"
        assert "85.8B" in doc.content

    def test_load_file_txt(self, tmp_path):
        f = tmp_path / "report.txt"
        f.write_text("Revenue: $100M\nProfit: $20M")
        doc = self.loader.load_file(str(f))
        assert doc.doc_id == "report"
        assert "$100M" in doc.content

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError):
            self.loader.load_file(str(f))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_file("/nonexistent/path.txt")

    def test_cleans_extra_whitespace(self):
        doc = self.loader.load_text("Revenue:   $100M\n\n\n\nProfit:   $20M", doc_id="d")
        assert "  " not in doc.content
        assert "\n\n\n" not in doc.content


# ── Chunker ─────────────────────────────────────────────────────────────────

class TestRecursiveChunker:

    def setup_method(self):
        from ingestion.loader import FinancialDocumentLoader, RecursiveChunker, Document
        self.chunker_cls = RecursiveChunker
        self.loader = FinancialDocumentLoader()
        self.Document = Document

    def _doc(self, text):
        return self.Document(doc_id="test", content=text, source="test")

    def test_short_text_single_chunk(self):
        chunker = self.chunker_cls(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_document(self._doc("Short text."))
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_long_text_multiple_chunks(self):
        text = "Revenue was $100M. " * 100
        chunker = self.chunker_cls(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(self._doc(text))
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        chunker = self.chunker_cls(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(self._doc("Apple " * 200))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        doc = self.loader.load_text("Q3 revenue...", doc_id="aapl", metadata={"ticker": "AAPL"})
        chunker = self.chunker_cls(chunk_size=500)
        chunks = chunker.chunk_document(doc)
        assert chunks[0].metadata["ticker"] == "AAPL"

    def test_token_count_estimate(self):
        chunker = self.chunker_cls()
        chunks = chunker.chunk_document(self._doc("word " * 100))
        assert all(c.token_count > 0 for c in chunks)


# ── Stub Embedder for tests (avoids downloading sentence-transformers) ────────

class StubEmbedder:
    """Deterministic stub — same interface as LocalEmbedder, no model download."""
    dim = 64

    def embed(self, texts):
        import hashlib
        vecs = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.vstack(vecs)

    def embed_query(self, q):
        return self.embed([q])[0]


# ── Vector Store ─────────────────────────────────────────────────────────────

class TestFAISSVectorStore:

    def setup_method(self):
        from retrieval.vector_store import FAISSVectorStore
        from ingestion.loader import Chunk
        self.Store = FAISSVectorStore
        self.Chunk = Chunk
        self.emb = StubEmbedder()

    def _chunk(self, i, text="test"):
        return self.Chunk(chunk_id=f"c{i}", doc_id="d", text=text, metadata={})

    def test_add_and_search(self):
        store = self.Store(dim=64)
        chunks = [self._chunk(i) for i in range(5)]
        vecs = self.emb.embed([c.text for c in chunks])
        store.add_chunks(chunks, vecs)
        q = vecs[0].copy()
        results = store.search(q, top_k=1)
        assert len(results) == 1

    def test_size(self):
        store = self.Store(dim=64)
        chunks = [self._chunk(i) for i in range(10)]
        store.add_chunks(chunks, self.emb.embed(["t"] * 10))
        assert store.size == 10

    def test_empty_search(self):
        store = self.Store(dim=64)
        assert store.search(np.zeros(64, dtype=np.float32)) == []

    def test_top_k(self):
        store = self.Store(dim=64)
        chunks = [self._chunk(i, f"text {i}") for i in range(20)]
        store.add_chunks(chunks, self.emb.embed([c.text for c in chunks]))
        results = store.search(np.random.randn(64).astype(np.float32), top_k=5)
        assert len(results) <= 5

    def test_mmr(self):
        store = self.Store(dim=64)
        chunks = [self._chunk(i, f"text {i}") for i in range(10)]
        store.add_chunks(chunks, self.emb.embed([c.text for c in chunks]))
        q = np.random.randn(64).astype(np.float32)
        results = store.search_mmr(q, top_k=4)
        assert len(results) == 4

    def test_save_load(self, tmp_path):
        store = self.Store(dim=64)
        store.add_chunks([self._chunk(0, "Apple earnings")],
                         self.emb.embed(["Apple earnings"]))
        store.save(str(tmp_path))

        store2 = self.Store(dim=64)
        store2.load(str(tmp_path))
        assert store2.size == 1
        assert store2._chunks[0].chunk_id == "c0"


# ── Generator ────────────────────────────────────────────────────────────────

class TestMockGenerator:

    def test_valid_json(self):
        import json
        from generation.generator import MockGenerator
        gen = MockGenerator()
        out = gen.generate("Apple revenue $85B.", "What was the revenue?")
        parsed = json.loads(out)
        assert "answer" in parsed and "confidence" in parsed

    def test_non_empty_answer(self):
        import json
        from generation.generator import MockGenerator
        out = MockGenerator().generate("Revenue: $100M", "Revenue?")
        assert len(json.loads(out)["answer"]) > 0


# ── Full Pipeline Integration ─────────────────────────────────────────────────

class TestPipelineIntegration:

    def _pipeline(self):
        """Build a pipeline with stub embedder to avoid model downloads in CI."""
        from pipeline import FinPulsePipeline
        from generation.generator import MockGenerator
        from generation.generator import FinPulseGenerator
        from retrieval.retriever import FinPulseRetriever
        from retrieval.vector_store import FAISSVectorStore

        p = FinPulsePipeline.__new__(FinPulsePipeline)
        p.top_k = 2
        p.retrieval_strategy = "similarity"
        from ingestion.loader import FinancialDocumentLoader, RecursiveChunker
        p.loader = FinancialDocumentLoader()
        p.chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
        p.embedder = StubEmbedder()
        p.store = FAISSVectorStore(dim=64)
        p.retriever = FinPulseRetriever(p.store, p.embedder)
        p.generator = FinPulseGenerator(llm=MockGenerator(), model="mock")
        return p

    def test_ingest_and_ask(self):
        p = self._pipeline()
        p.ingest_text(
            "Apple Q3 2024 revenue was $85.8 billion, up 5% YoY. "
            "iPhone revenue was $39.3 billion. Services hit a record $24.2 billion.",
            doc_id="aapl_q3",
        )
        answer = p.ask("What was Apple's revenue?")
        assert isinstance(answer.answer, str) and len(answer.answer) > 0

    def test_raises_without_documents(self):
        from pipeline import FinPulsePipeline
        p = self._pipeline()
        with pytest.raises(RuntimeError, match="No documents indexed"):
            p.ask("What is the revenue?")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
