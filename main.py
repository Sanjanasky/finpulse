"""
FinPulse RAG — Demo (fully local, free)

Quick start:
  pip install -r requirements.txt
  ollama pull llama3.2          # one-time ~2GB download
  python main.py

No API keys needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import FinPulsePipeline


BANNER = """
╔══════════════════════════════════════════════════════╗
║         FinPulse RAG  —  Local & Free Mode          ║
║   Embeddings: sentence-transformers (CPU)           ║
║   LLM:        Ollama (llama3.2 / mistral / phi3)    ║
╚══════════════════════════════════════════════════════╝
"""

QUESTIONS = [
    "What was Apple's total revenue in Q3 2024?",
    "How did the Services segment perform compared to last year?",
    "What is Apple's revenue guidance for Q4 2024?",
    "What is Apple's cash position and how much did they buy back in shares?",
    "What risks exist for Apple in Greater China?",
]


def run():
    print(BANNER)

    pipeline = FinPulsePipeline(
        chunk_size=400,
        chunk_overlap=60,
        top_k=4,
        retrieval_strategy="mmr",
        ollama_model="llama3.2",   # change to mistral / phi3 / qwen2.5 etc.
        # force_mock=True,         # uncomment to test without Ollama
    )

    n = pipeline.ingest_directory("data/sample_docs")
    if n == 0:
        print("Add .txt/.md/.pdf files to data/sample_docs/ and re-run.")
        return

    for question in QUESTIONS:
        print(f"\n{'─' * 60}")
        answer = pipeline.ask(question)
        print()
        print(answer.display())

    pipeline.save("./finpulse_index")
    print(f"\n\n💾 Index saved. Next run, skip re-embedding:")
    print('   pipeline.load("./finpulse_index")')


if __name__ == "__main__":
    run()
