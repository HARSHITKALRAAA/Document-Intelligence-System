"""Streamlit interface for ContractLens."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from contractlens.answer import FIELDS, answer_question, configured_key
from contractlens.core import VectorIndex, chunk_pages, extract_pdf_pages

load_dotenv(Path(__file__).with_name(".env"))


@st.cache_resource(show_spinner="Loading the local embedding model...")
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def uploaded_set(files):
    """Use content digests to ensure an old index is never used for new PDFs."""
    return [(file.name, file.getvalue()) for file in files]


def signature(items):
    digest = hashlib.sha256()
    for name, data in items:
        digest.update(name.encode())
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def build_index(items):
    pages = []
    for name, content in items:
        pages.extend(extract_pdf_pages(name, content))
    passages = chunk_pages(pages)
    return VectorIndex(passages, embedding_model()), len(pages)


st.set_page_config(page_title="ContractLens", page_icon="📑", layout="wide")
st.title("📑 ContractLens")
st.caption("Ask questions about your contracts and inspect the supporting pages.")

for key, initial in (("index", None), ("signature", None), ("history", []), ("terms", None), ("term_scope", None)):
    if key not in st.session_state:
        st.session_state[key] = initial

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload text-based PDF contracts", type=["pdf"], accept_multiple_files=True)
    st.caption("Files stay in this app session. Text used for an answer is sent to Groq.")
    api_key = configured_key()
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Only needed to generate answers.")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    process = st.button("Process documents", type="primary", disabled=not uploads)

items = uploaded_set(uploads or [])
current_signature = signature(items) if items else None
if current_signature != st.session_state.signature:
    st.session_state.index = None
    st.session_state.history = []
    st.session_state.terms = None
    st.session_state.signature = current_signature

if process:
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        st.error("Two files have the same name. Rename one before uploading.")
    elif any(len(data) > 15 * 1024 * 1024 for _, data in items):
        st.error("Each PDF must be 15 MB or smaller.")
    else:
        try:
            with st.spinner("Extracting pages and embedding passages..."):
                index, page_count = build_index(items)
            st.session_state.index = index
            st.success(f"Indexed {len(items)} PDFs, {page_count} pages, and {len(index.passages)} passages.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

index = st.session_state.index
if index is None:
    st.info("Upload one or more text-based PDFs and select Process documents.")
    st.stop()

names = sorted({p.document for p in index.passages})
selected = st.multiselect("Search these contracts", names, default=names)
if st.session_state.term_scope != tuple(selected):
    st.session_state.terms = None
    st.session_state.term_scope = tuple(selected)
question = st.chat_input("For example: What is the termination notice period?")

if question:
    if not selected:
        st.warning("Select at least one contract.")
    else:
        hits = index.search(question, top_k=5, documents=set(selected))
        try:
            with st.spinner("Checking the relevant passages..."):
                answer = answer_question(question, hits, api_key, model)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            answer = f"Answer unavailable: {exc}"
        st.session_state.history.append((question, answer, hits))

for asked, answer, hits in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(asked)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Retrieved sources and similarity scores"):
            for i, hit in enumerate(hits, 1):
                st.markdown(f"**[S{i}] {hit.passage.document} · page {hit.passage.page} · score {hit.score:.2f}**")
                st.write(hit.passage.text)

st.divider()
st.subheader("Key contract terms")
st.caption("Extracts five fields from the selected documents. Check each cited page in the PDF before relying on a field.")
if st.button("Extract key terms", disabled=not selected):
    terms = {}
    try:
        with st.spinner("Extracting contract terms..."):
            for label, prompt in FIELDS.items():
                hits = index.search(prompt, top_k=4, documents=set(selected))
                terms[label] = answer_question(prompt, hits, api_key, model)
        st.session_state.terms = terms
    except (ValueError, RuntimeError, KeyError, IndexError) as exc:
        st.error(str(exc))
if st.session_state.terms:
    for label, value in st.session_state.terms.items():
        st.markdown(f"**{label}:** {value}")

st.caption("Document analysis aid. Verify extracted terms against the original PDF.")
