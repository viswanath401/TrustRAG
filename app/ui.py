
import sys
import re
from pathlib import Path
import tempfile
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.hybrid_search import HybridSearch
from app.reranker import rerank_results
from app.llm import generate_answer
from app.citation import verify_citations
from app.confidence import calculate_confidence


st.set_page_config(
    page_title="TrustRAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 TrustRAG")
st.caption("Hybrid Multi-Agent Research Assistant")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("📄 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Papers", use_container_width=True):
        if uploaded_files:
            all_chunks = []

            with st.spinner("Processing research papers..."):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp:
                        temp.write(file.getvalue())
                        path = temp.name

                    try:
                        pages = extract_text(path)
                        chunks = create_chunks(pages)

                        for chunk in chunks:
                            chunk["source"] = file.name

                        all_chunks.extend(chunks)

                    finally:
                        os.remove(path)

            st.session_state.chunks = all_chunks
            st.session_state.history = []

            st.success(
                f"{len(all_chunks)} chunks processed successfully!"
            )
        else:
            st.warning("Please upload a PDF first.")

    st.divider()

    st.metric(
        "Total Chunks",
        len(st.session_state.chunks)
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.subheader("💬 Ask Your Research Question")

query = st.chat_input(
    "Ask something about your uploaded research papers..."
)

if query:
    if not st.session_state.chunks:
        st.warning("Upload and process research papers first.")

    else:
        with st.spinner("TrustRAG is researching your question..."):

            hybrid = HybridSearch(
                st.session_state.chunks
            )

            results = hybrid.search(
                query,
                top_k=10
            )

            ranked = rerank_results(
                query,
                results,
                top_k=5
            )

            answer = generate_answer(
                query,
                ranked
            )

            citation_report = verify_citations(
                answer,
                ranked
            )

            confidence = calculate_confidence(
                ranked,
                citation_report
            )

        st.session_state.history.append({
            "question": query,
            "answer": answer,
            "citations": citation_report,
            "confidence": confidence,
            "sources": ranked
        })


for item in st.session_state.history:

    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):

        answer = item["answer"]

        # Fix HTML line breaks in generated Markdown
        answer = re.sub(
            r"<br\s*/?>",
            " — ",
            answer,
            flags=re.IGNORECASE
        )

        st.markdown(answer)

        st.divider()

        st.subheader("📊 Confidence Report")

        confidence = item["confidence"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence Score",
                f'{confidence["confidence_score"]}/100'
            )

        with col2:
            st.metric(
                "Confidence Level",
                confidence["level"]
            )

        st.progress(
            min(
                max(confidence["confidence_score"] / 100, 0),
                1
            )
        )

        with st.expander("View Confidence Details"):
            st.json(confidence)

        with st.expander("Citation Verification"):
            st.json(item["citations"])

        with st.expander("Retrieved Research Evidence"):

            for i, source in enumerate(item["sources"]):

                st.markdown(
                    f"**Source {i+1} — Page {source['page']}**"
                )

                st.caption(
                    f"Reranking score: "
                    f"{source['rerank_score']:.4f}"
                )

                st.write(source["text"])

                st.divider()
