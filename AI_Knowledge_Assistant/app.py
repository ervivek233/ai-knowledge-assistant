# app.py
import os
import io
import time
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()  # this will read .env file
api_key = os.getenv("PERPLEXITY_API_KEY")

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="RAG Chatbot (Perplexity)", page_icon="📚", layout="wide")
st.title("📚 RAG Chatbot (FAISS + Sentence Transformers + Perplexity)")
st.write("Upload PDFs/TXT, build the index, and ask questions. Answers are grounded in retrieved context with citations.")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("PERPLEXITY_API_KEY", os.getenv("PERPLEXITY_API_KEY", ""), type="password")
    # Common Perplexity models
    model_name = st.selectbox(
    "Model",
    options=[
        "sonar",                 # general-purpose, 50 RPM
        "sonar-pro",             # advanced features
        "sonar-reasoning",       # reasoning-focused
        "sonar-reasoning-pro",   # reasoning + pro features
        "sonar-deep-research",   # slower, deep research
    ],
    index=0
)
    top_k = st.slider("Top‑K retrieved chunks", 2, 10, 4)
    max_tokens = st.slider("Max tokens (response)", 128, 4096, 1024, step=64)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.markdown("**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local)")
    st.caption("Embeddings computed locally; no external calls for retrieval.")

if not api_key:
    st.warning("Enter your PERPLEXITY_API_KEY in the sidebar to enable generation.")

# --------------------------------------------------
# Clients & models
# --------------------------------------------------
client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai") if api_key else None

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# --------------------------------------------------
# Ingestion helpers
# --------------------------------------------------
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            texts.append(txt)
    return "\n".join(texts)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def chunk_texts(raw_text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(raw_text)

def build_corpus(files: List[Any]) -> List[Dict[str, Any]]:
    corpus = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        data = f.read()
        text = read_pdf(data) if name.lower().endswith(".pdf") else read_txt(data)
        if text.strip():
            chunks = chunk_texts(text)
            for i, ch in enumerate(chunks):
                corpus.append({"text": ch, "source": f"{name}#chunk{i+1}"})
    return corpus

# --------------------------------------------------
# FAISS vector store
# --------------------------------------------------
class FaissStore:
    def __init__(self, dim: int = EMBED_DIM):
        # cosine similarity via normalized inner product
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []
        self.embeds = None

    def add(self, docs: List[Dict[str, Any]]):
        texts = [d["text"] for d in docs]
        X = embedder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        self.index.add(Xn.astype("float32"))
        self.meta.extend(docs)
        self.embeds = Xn if self.embeds is None else np.vstack([self.embeds, Xn])

    def search(self, query: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
        q = embedder.encode([query], convert_to_numpy=True)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(qn.astype("float32"), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[idx]))
        return results

# --------------------------------------------------
# Prompt assembly & generation
# --------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context.\n"
    "Rules:\n"
    "- Use the CONTEXT strictly; if the answer is not present, say you don't find it.\n"
    "- Be concise and accurate.\n"
    "- Cite sources by their labels (e.g., file#chunkN) at the end.\n"
)

def build_messages(contexts: List[Dict[str, Any]], user_query: str) -> List[Dict[str, str]]:
    context_block = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in contexts])
    user_block = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{user_query}\n\n"
        "Return a direct answer. If unsure or not in context, state that clearly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

def generate_answer(messages: List[Dict[str, str]]) -> str:
    if client is None:
        return "API key missing. Please add PERPLEXITY_API_KEY in the sidebar."
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# --------------------------------------------------
# UI state
# --------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dict: {user, assistant, sources}

# --------------------------------------------------
# App UI
# --------------------------------------------------
uploaded_files = st.file_uploader("Upload PDFs/TXT", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔧 Build / Rebuild Index", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one file.")
        else:
            with st.spinner("Processing documents and building FAISS index..."):
                corpus = build_corpus(uploaded_files)
                if not corpus:
                    st.error("No readable text found.")
                else:
                    store = FaissStore()
                    store.add(corpus)
                    st.session_state.store = store
                    st.success(f"Indexed {len(corpus)} chunks from {len(uploaded_files)} file(s).")

with col2:
    st.button("🗑️ Clear Chat", on_click=lambda: st.session_state.update({"chat": []}))

st.divider()

query = st.text_input("Ask a question about your documents:")
go = st.button("Ask")

if go and query.strip():
    if st.session_state.store is None:
        st.warning("Build the index first.")
    elif client is None:
        st.warning("Add your PERPLEXITY_API_KEY.")
    else:
        with st.spinner("Retrieving relevant context..."):
            results = st.session_state.store.search(query, k=top_k)
            contexts = [r[1] for r in results]
        with st.spinner("Generating answer..."):
            try:
                messages = build_messages(contexts, query)
                answer = generate_answer(messages)
            except Exception as e:
                st.error(f"Generation error: {e}")
                answer = "There was an error generating a response."
        st.session_state.chat.append(
            {"user": query, "assistant": answer, "sources": [c["source"] for c in contexts]}
        )

# --------------------------------------------------
# Chat rendering with enhancements
# --------------------------------------------------
if st.session_state.chat:
    st.subheader("Chat")
    for turn in st.session_state.chat:
        # User message
        with st.chat_message("user"):
            st.write(turn["user"])

        # Assistant message
        with st.chat_message("assistant"):
            st.write(turn["assistant"])

            if turn.get("sources"):
                # Make sources clickable
                st.caption("Sources: " + " • ".join([f"[{s}](#{s})" for s in turn["sources"]]))

                # Show retrieved chunks inline for source highlighting
                for src in turn["sources"]:
                    for doc in st.session_state.store.meta:
                        if doc["source"] == src:
                            st.markdown(f"**{src}**")  # filename + chunk number
                            st.code(doc["text"][:500])  # show first 500 chars of chunk

# --------------------------------------------------
# Context preview
# --------------------------------------------------
with st.expander("🔎 Retrieved context (latest question)"):
    if not st.session_state.chat:
        st.info("No queries yet.")
    else:
        last_sources = st.session_state.chat[-1].get("sources", [])
        store = st.session_state.store
        if store is None:
            st.info("Index not built.")
        elif not last_sources:
            st.info("No context retrieved yet.")
        else:
            st.write("Showing text of the retrieved chunks:")
            shown = 0
            for src in last_sources:
                for doc in store.meta:
                    if doc["source"] == src:
                        st.markdown(f"**{src}**")
                        st.code(doc["text"][:2000])
                        shown += 1
                        break
            if shown == 0:
                st.info("Could not find retrieved chunks in memory.")

st.divider()
st.caption("Built with FAISS + Sentence Transformers for retrieval, and Perplexity API for generation.")
