from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import hashlib
import tempfile
import os

MAX_PAGES = 50  # Max pages to process — keeps it under 10 seconds

def get_file_hash(uploaded_files):
    hasher = hashlib.md5()
    for f in uploaded_files:
        hasher.update(f.name.encode())
        hasher.update(str(f.size).encode())
    return hasher.hexdigest()

@st.cache_resource(show_spinner=False)
def build_cached_vector_store(_cache_key, chunks_text):
    """
    Cache vector store by file hash.
    Same file = instant reload, no reprocessing.
    chunks_text is a tuple of strings for hashing purposes.
    """
    from langchain_core.documents import Document
    chunks = [Document(page_content=t, metadata=m) for t, m in chunks_text]
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_documents(chunks, embeddings)

def load_and_chunk_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)

        if len(pages) > MAX_PAGES:
            pages = pages[:MAX_PAGES]

        for page in pages:
            page.metadata["source_file"] = uploaded_file.name

        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(all_docs)

def build_qa_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1024
    )

    prompt = ChatPromptTemplate.from_template("""You are a helpful study assistant for GATE and JEE exam preparation.
Use ONLY the context below to answer the question. If the answer is not in the context,
say "I couldn't find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:""")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever