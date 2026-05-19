# 🏦 FinPulse RAG Pipeline

> **Ask questions about financial documents — fully local, completely free, no API keys.**

FinPulse is a production-ready **Retrieval-Augmented Generation (RAG)** pipeline built for financial documents. Upload earnings reports, 10-Ks, analyst notes, or any financial text — then ask questions in plain English and get accurate, cited answers powered by a local LLM running on your own machine.

---

## ✨ Features

- 🔒 **100% Local** — nothing leaves your machine
- 🆓 **Completely Free** — no OpenAI, no Anthropic, no API bills
- 🧠 **Smart Retrieval** — MMR search returns diverse, relevant chunks
- 📄 **Multi-format** — supports `.txt`, `.pdf`, `.md` documents
- 💬 **Chat UI** — beautiful Gradio interface in your browser
- 💾 **Persistent Index** — save and reload your vector index
- ✅ **Tested** — 20 unit + integration tests

---

## 🖥️ Demo

```
❓ What was Apple's total revenue in Q3 2024?
💬 Apple's total revenue in Q3 2024 was $85.8 billion [Source 1],
   representing a 5% increase year-over-year.
📊 Confidence: HIGH | 🤖 Model: llama3.2
```

---

## 🏗️ Architecture

```
┌─────────────────────── INGESTION ───────────────────────┐
│  Documents (.pdf / .txt / .md)                          │
│          ↓                                              │
│  FinancialDocumentLoader → RecursiveChunker             │
│          ↓                                              │
│  LocalEmbedder  (sentence-transformers, CPU, free)      │
│          ↓                                              │
│  FAISSVectorStore  (IndexFlatIP, cosine similarity)     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────── QUERY ───────────────────────────┐
│  User Question                                          │
│          ↓                                              │
│  LocalEmbedder (same model, embed query)                │
│          ↓                                              │
│  MMR Search (Maximal Marginal Relevance, top-k chunks)  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────── GENERATION ──────────────────────┐
│  Retrieved Chunks + Question                            │
│          ↓                                              │
│  Prompt Builder (financial analyst system prompt)       │
│          ↓                                              │
│  Ollama LLM  (llama3.2 / mistral / phi3 / qwen2.5)     │
│          ↓                                              │
│  JSON Output Parser → FinPulseAnswer                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Sanjanasky/finpulse.git
cd finpulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama + pull a free model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com/download

# Pull a model (choose one)
ollama pull llama3.2      # ⭐ Best overall — 2GB
ollama pull phi3          # Lightweight — works on 4GB RAM
ollama pull mistral       # Great JSON output — 4GB
ollama pull qwen2.5       # Best for finance/math — 4.7GB
```

### 4. Add your financial documents

```bash
# Drop any .txt / .pdf / .md files into:
data/sample_docs/
```

### 5. Run the terminal version

```bash
python main.py
```

### 6. Or run the browser UI

```bash
python app.py
# Opens at http://localhost:7860
```

---

## 🖼️ Gradio UI

Run `python app.py` and open `http://localhost:7860` in your browser.

```
┌─────────────────────────────────────────────┐
│  🏦 FinPulse RAG                            │
│  [ 🚀 Initialise Pipeline ]  [ Upload Doc ] │
│                                             │
│  💬 Chat window                             │
│  ─────────────────────────────────────────  │
│  [ Ask a financial question...    ] [ Ask ] │
│                                             │
│  💡 Apple Q3 revenue? | Services YoY? ...  │
└─────────────────────────────────────────────┘
```

1. Click **🚀 Initialise Pipeline** to load documents
2. Type a question or click a suggestion
3. Upload new documents anytime — they're indexed instantly

---

## 📁 Project Structure

```
finpulse/
│
├── ingestion/
│   ├── loader.py          # FinancialDocumentLoader + RecursiveChunker
│   └── embedder.py        # LocalEmbedder (sentence-transformers)
│
├── retrieval/
│   ├── vector_store.py    # FAISSVectorStore — cosine + MMR search
│   └── retriever.py       # FinPulseRetriever
│
├── generation/
│   └── generator.py       # OllamaGenerator + FinPulseGenerator
│
├── data/
│   └── sample_docs/       # 📂 Put your financial docs here
│
├── app.py                 # Gradio web UI
├── pipeline.py            # Main pipeline orchestrator
├── main.py                # Terminal demo
├── tests.py               # 20 unit + integration tests
├── requirements.txt
└── .env.example
```

---

## 💻 Usage in Code

```python
from pipeline import FinPulsePipeline

# Initialise
pipeline = FinPulsePipeline(
    chunk_size=400,
    chunk_overlap=60,
    top_k=4,
    retrieval_strategy="mmr",
    ollama_model="llama3.2",
)

# Ingest documents
pipeline.ingest_directory("data/sample_docs")

# Ask a question
answer = pipeline.ask("What was Apple's Q3 2024 revenue?")
print(answer.display())

# Save index — skip re-embedding next run
pipeline.save("./finpulse_index")

# Next run — load saved index
pipeline.load("./finpulse_index")
```

### Answer object

```python
answer.answer                # Full answer with [Source N] citations
answer.confidence            # "high" | "medium" | "low"
answer.sources_used          # [1, 2] — which chunks were cited
answer.caveats               # Limitations flagged by the LLM
answer.follow_up_questions   # Suggested next questions
answer.retrieved_chunks      # Raw retrieved chunks with scores
answer.model_used            # Which Ollama model answered
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 400 | Characters per chunk |
| `chunk_overlap` | 60 | Overlap between chunks |
| `top_k` | 4 | Chunks retrieved per query |
| `retrieval_strategy` | `mmr` | `mmr` (diverse) or `similarity` |
| `embedding_model` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `ollama_model` | `llama3.2` | Any model from `ollama list` |

### Free embedding models

| Model | Size | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 22MB | Default — fast, good quality |
| `BAAI/bge-small-en-v1.5` | 33MB | Better accuracy |
| `BAAI/bge-base-en-v1.5` | 110MB | Best quality on CPU |

---

## 🧪 Running Tests

```bash
python -m pytest tests.py -v
```

20 tests — runs fully offline, no Ollama needed.

---

## 📦 Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) for LLM generation
- 4GB RAM minimum (8GB recommended)
- No GPU needed — runs entirely on CPU

---

## 🔧 Troubleshooting

**Ollama not found?**
```bash
ollama serve        # start Ollama manually
ollama pull llama3.2
```

**sentence-transformers error?**
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.0 sentence-transformers==3.0.1
```

**Gradio won't open on localhost?**
```python
# In app.py, change the last line to:
demo.launch(share=True)
```

---

## 🗺️ Roadmap

- [ ] Multi-document comparison queries
- [ ] Table extraction from PDFs
- [ ] Streaming answers in Gradio UI
- [ ] Support for Excel / CSV financial data
- [ ] Docker container for one-command setup

---

## 📄 License

MIT — free for personal and commercial use.

---

## 🙏 Built With

- [Ollama](https://ollama.com) — local LLM runtime
- [sentence-transformers](https://www.sbert.net/) — free embeddings
- [FAISS](https://faiss.ai/) — vector similarity search
- [Gradio](https://gradio.app/) — browser UI
