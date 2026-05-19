"""
FinPulse RAG — Gradio Web UI
Run: python app.py
Opens at: http://localhost:7860
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from pipeline import FinPulsePipeline

# ── Global pipeline instance ─────────────────────────────────────────────────
pipeline = None


def init_pipeline():
    global pipeline
    pipeline = FinPulsePipeline(
        chunk_size=400,
        chunk_overlap=60,
        top_k=4,
        retrieval_strategy="mmr",
        ollama_model="llama3.2",
        # force_mock=True,  # uncomment if Ollama not installed yet
    )
    pipeline.ingest_directory("data/sample_docs")
    return "✅ Pipeline ready! Ask your financial questions below."


def ask_question(question, history):
    global pipeline

    if pipeline is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Pipeline not initialised. Click **Initialise Pipeline** first."})
        return history

    if not question.strip():
        return history

    try:
        answer = pipeline.ask(question)

        response = f"**{answer.answer}**\n\n"
        response += f"📊 Confidence: `{answer.confidence.upper()}` | 🤖 Model: `{answer.model_used}`\n\n"

        if answer.caveats:
            response += f"⚠️ *{answer.caveats}*\n\n"

        if answer.follow_up_questions:
            response += "🔎 **You might also ask:**\n"
            for q in answer.follow_up_questions:
                response += f"- {q}\n"

        response += "\n\n---\n📄 **Retrieved Sources:**\n"
        for r in answer.retrieved_chunks:
            source = r.chunk.metadata.get("filename", r.chunk.doc_id)
            response += f"\n**[Source {r.rank}]** `{source}` — score: `{r.score:.3f}`\n"
            response += f"> {r.chunk.text[:200]}...\n"

    except Exception as e:
        response = f"❌ Error: {str(e)}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    return history


def upload_document(file):
    global pipeline
    if pipeline is None:
        return "⚠️ Initialise the pipeline first."
    if file is None:
        return "No file uploaded."
    try:
        n = pipeline.ingest_file(file)
        return f"✅ Indexed {n} chunks from `{os.path.basename(file)}`"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def set_question(q):
    return q


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="FinPulse RAG") as demo:

    gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1>🏦 FinPulse RAG</h1>
            <p style="color:#666;">Financial Document Q&A — Local & Free (Ollama + sentence-transformers)</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            init_btn = gr.Button("🚀 Initialise Pipeline", variant="primary", size="lg")
            status = gr.Textbox(
                label="Status",
                value="Click Initialise Pipeline to start...",
                interactive=False,
            )
        with gr.Column(scale=1):
            upload = gr.File(
                label="📄 Upload a financial document (.txt / .pdf / .md)",
                file_types=[".txt", ".pdf", ".md"],
            )
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

    chatbot = gr.Chatbot(
        label="FinPulse Chat",
        height=480,
        type="messages",
        show_copy_button=True,
    )

    with gr.Row():
        question = gr.Textbox(
            placeholder="Ask a financial question... e.g. What was Apple's Q3 2024 revenue?",
            label="Your Question",
            scale=5,
        )
        ask_btn = gr.Button("Ask ➤", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

    gr.Markdown("### 💡 Try these questions:")
    with gr.Row():
        for q in [
            "What was Apple's total revenue in Q3 2024?",
            "How did Services perform year over year?",
            "What is Apple's Q4 2024 guidance?",
            "What is Apple's cash position?",
        ]:
            gr.Button(q, size="sm").click(fn=lambda x=q: x, outputs=question)

    # Events
    init_btn.click(fn=init_pipeline, outputs=status)

    ask_btn.click(
        fn=ask_question,
        inputs=[question, chatbot],
        outputs=chatbot,
    ).then(fn=lambda: "", outputs=question)

    question.submit(
        fn=ask_question,
        inputs=[question, chatbot],
        outputs=chatbot,
    ).then(fn=lambda: "", outputs=question)

    upload.change(fn=upload_document, inputs=upload, outputs=upload_status)
    clear_btn.click(fn=lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
