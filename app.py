import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import (
    load_and_chunk_pdfs, build_vector_store,
    build_qa_chain, get_file_hash, MAX_PAGES
)

load_dotenv()

st.set_page_config(
    page_title="GATE & JEE Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #16213e 100%);
    border-right: 1px solid #3d3d7a;
}
h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 10px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: transform 0.2s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03);
    border: 2px dashed #3d3d7a;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid #3d3d7a;
    border-radius: 10px;
}
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

MAX_HISTORY = 20
MAX_FILE_SIZE_MB = 10

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        groq_api_key = ""

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found.")
    st.stop()

st.title("📚 GATE & JEE Study Assistant")
st.caption("Upload past year papers or study material → Ask questions → Get instant answers with sources")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📂 Upload PDFs")
    st.caption(f"⚠️ Max file size: {MAX_FILE_SIZE_MB}MB | Max pages processed: {MAX_PAGES}")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Upload PDFs under {MAX_FILE_SIZE_MB}MB each."
    )

    if uploaded_files:
        oversized = []
        valid_files = []
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                oversized.append(f"❌ {f.name} ({size_mb:.1f}MB) — too large")
            else:
                valid_files.append(f)

        if oversized:
            st.error(
                f"Files over {MAX_FILE_SIZE_MB}MB will be skipped:\n\n"
                + "\n".join(oversized)
                + "\n\n💡 Tip: Split large PDFs at ilovepdf.com"
            )

        if valid_files:
            total_mb = sum(f.size for f in valid_files) / (1024 * 1024)
            st.info(f"📄 {len(valid_files)} file(s) ready — {total_mb:.1f}MB total")

            if st.button("⚡ Process PDFs", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Reading PDFs...")

                progress_bar.progress(25, text="📖 Loading pages...")
                chunks = load_and_chunk_pdfs(valid_files)

                progress_bar.progress(60, text="🧠 Building embeddings...")
                vector_store = build_vector_store(chunks)
                st.session_state.vector_store = vector_store

                progress_bar.progress(85, text="⚙️ Setting up QA chain...")
                st.session_state.qa_chain, st.session_state.retriever = build_qa_chain(
                    vector_store, groq_api_key
                )
                st.session_state.chat_history = []

                progress_bar.progress(100, text="✅ Done!")

                col1, col2 = st.columns(2)
                col1.metric("📄 Chunks Created", len(chunks))
                col2.metric("📁 PDFs Indexed", len(valid_files))
                st.success("✅ Ready! Ask your first question →")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown(
        f"1. Upload PDFs (max {MAX_FILE_SIZE_MB}MB each)\n"
        f"2. Only first {MAX_PAGES} pages indexed\n"
        "3. Click Process\n"
        "4. Ask any question\n"
        "5. See answer + source page"
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


def format_page_number(page_num):
    try:
        return str(int(page_num) + 1)
    except (ValueError, TypeError):
        return "?"


if st.session_state.qa_chain is None:
    st.info("👈 Upload and process your PDFs from the sidebar to start asking questions.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📖 GATE Papers**")
        st.caption("Upload GATE CS/EC/ME previous year papers and ask topic-wise questions")
    with col2:
        st.markdown("**🧮 JEE Material**")
        st.caption("Upload NCERT, HC Verma, or JEE Advanced papers and quiz yourself")
    with col3:
        st.markdown("**📝 Any Study Material**")
        st.caption("Works with any PDF — upload specific chapters for best results")

else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 Source pages used to answer"):
                    for i, source in enumerate(message["sources"], 1):
                        file_name = source.metadata.get("source_file", "Unknown file")
                        page_num = source.metadata.get("page", "?")
                        st.markdown(f"**Source {i}:** `{file_name}` — Page {format_page_number(page_num)}")
                        st.caption(source.page_content[:300] + "...")

    if question := st.chat_input("Ask a question about your uploaded documents..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                source_docs = st.session_state.retriever.invoke(question)
                answer = st.session_state.qa_chain.invoke(question)

            st.markdown(answer)

            with st.expander("📄 Source pages used to answer"):
                for i, source in enumerate(source_docs, 1):
                    file_name = source.metadata.get("source_file", "Unknown file")
                    page_num = source.metadata.get("page", "?")
                    st.markdown(f"**Source {i}:** `{file_name}` — Page {format_page_number(page_num)}")
                    st.caption(source.page_content[:300] + "...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs
        })

        if len(st.session_state.chat_history) > MAX_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]