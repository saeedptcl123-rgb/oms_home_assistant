"""
OMS Policy Assistant — RAG chat app
Loads a prebuilt FAISS index, retrieves relevant chunks for a user question,
and sends them to Groq (openai/gpt-oss-120b) to generate a grounded answer.
"""

import os
import pickle

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------- Config ----------
INDEX_DIR = "faiss_index"
MODEL_NAME = "openai/gpt-oss-120b"   # Groq-hosted model
TOP_K = 4

st.set_page_config(page_title="OMS Policy Assistant", page_icon="📄", layout="centered")


# ---------- Load index + data (cached so it only runs once) ----------
@st.cache_resource
def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))

    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    with open(os.path.join(INDEX_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    embed_model = SentenceTransformer(config["embed_model_name"])

    return index, chunks, metadata, embed_model


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Add it to `.streamlit/secrets.toml`.")
        st.stop()
    return Groq(api_key=api_key)


def retrieve(query, index, chunks, metadata, embed_model, top_k=TOP_K):
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx],
            "meta": metadata[idx],
            "score": float(dist),
        })
    return results


def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for i, r in enumerate(retrieved_chunks, 1):
        m = r["meta"]
        context_blocks.append(
            f"[Source {i}: {m['source_file']} (Dept: {m['department']}, Page: {m['page']})]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an OMS policy assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not in the context, say you don't "
        "have enough information in the knowledge base to answer. Be concise and clear. "
        "When relevant, refer to the source document by name."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    return system_prompt, user_prompt


def generate_answer(client, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.title("📄 OMS Policy Assistant")
st.caption("Ask a question about OMS policies. Answers are grounded in your uploaded documents.")

index, chunks, metadata, embed_model = load_index()
client = load_groq_client()

if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_file']}** — {s['department']} (page {s['page']})")

# Chat input
question = st.chat_input("Ask a question about OMS policy...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer..."):
            retrieved = retrieve(question, index, chunks, metadata, embed_model)

            if not retrieved:
                answer = "I couldn't find anything relevant in the knowledge base."
                sources = []
            else:
                system_prompt, user_prompt = build_prompt(question, retrieved)
                answer = generate_answer(client, system_prompt, user_prompt)
                sources = [r["meta"] for r in retrieved]

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['source_file']}** — {s['department']} (page {s['page']})")

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
