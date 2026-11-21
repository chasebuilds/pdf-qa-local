"""Optional Streamlit UI. Run: streamlit run src/pdfqa/ui.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pdfqa.cli import DEFAULT_DB, format_answer, make_embedder, make_llm, open_retriever
from pdfqa.qa import ask
from pdfqa.store import Index

st.set_page_config(page_title="pdf-qa-local", page_icon="📄")
st.title("pdf-qa-local")
st.caption("Questions about your PDFs. Everything runs on this machine.")

stub = st.sidebar.checkbox("Stub mode (no Ollama)", value=False)
db = Path(st.sidebar.text_input("Index file", str(DEFAULT_DB)))
folder = st.sidebar.text_input("PDF folder", "docs")

if st.sidebar.button("Index folder"):
    idx = Index(db)
    try:
        s = idx.index_folder(Path(folder), make_embedder(stub))
        st.sidebar.success(f"{s.files_indexed} files indexed, {s.chunks_added} chunks")
    except (ValueError, OSError, RuntimeError) as e:
        st.sidebar.error(str(e))
    finally:
        idx.close()

question = st.text_input("Ask a question")
if question:
    try:
        retriever = open_retriever(db, make_embedder(stub))
        ans = ask(question, retriever, make_llm(stub))
        st.markdown(format_answer(ans))
        with st.expander("Retrieved passages"):
            for h in ans.hits:
                st.markdown(
                    f"**{h.chunk.file} p.{h.chunk.page}** "
                    f"(cosine {h.cosine:.2f}, keyword coverage {h.coverage:.0%})"
                )
                st.write(h.chunk.text)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
