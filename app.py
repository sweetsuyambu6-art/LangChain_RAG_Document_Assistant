import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
st.set_page_config(page_title="LangChain RAG Assistant", page_icon="🔎", layout="wide")
st.title("🔎 LangChain RAG Document Assistant")
st.write("Upload documents and ask questions. Answers are generated from retrieved passages.")

with st.sidebar:
    st.header("RAG settings")
    model_name = st.text_input("Ollama chat model", os.getenv("OLLAMA_MODEL", "llama3.2"))
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    top_k = st.slider("Retrieved chunks (k)", 1, 8, 4)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def read_uploaded_file(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded.getvalue())
        temp_path = temp.name
    try:
        if suffix == ".pdf":
            documents = PyPDFLoader(temp_path).load()
        elif suffix == ".docx":
            documents = Docx2txtLoader(temp_path).load()
        elif suffix == ".txt":
            documents = TextLoader(temp_path, encoding="utf-8").load()
        else:
            documents = []
        for document in documents:
            document.metadata["source"] = uploaded.name
        return documents
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

uploads = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Index documents", type="primary"):
    if not uploads:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Loading documents and building the vector index..."):
            all_documents = []
            for uploaded in uploads:
                try:
                    all_documents.extend(read_uploaded_file(uploaded))
                except Exception as error:
                    st.error(f"Could not read {uploaded.name}: {error}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=min(chunk_overlap, chunk_size - 1),
                add_start_index=True,
            )
            chunks = splitter.split_documents(all_documents)
            if not chunks:
                st.error("No readable text was found in the uploaded files.")
            else:
                try:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=get_embeddings(),
                        collection_name="langchain_rag_docs",
                    )
                    st.session_state.indexed_names = [file.name for file in uploads]
                    st.success(
                        f"Indexed {len(all_documents)} pages/sections into {len(chunks)} chunks."
                    )
                except Exception as error:
                    st.error(f"Could not create vector index: {error}")

if st.session_state.indexed_names:
    st.caption("Indexed files: " + ", ".join(st.session_state.indexed_names))

question = st.text_input("Question", placeholder="What does the uploaded document say about ...?")
if st.button("Get answer"):
    if st.session_state.vector_store is None:
        st.warning("Index your documents before asking a question.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant text and generating an answer..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": top_k}
                )
                retrieved_docs = retriever.invoke(question)

                context_blocks = []
                for index, document in enumerate(retrieved_docs, start=1):
                    source = document.metadata.get("source", "Unknown source")
                    page = document.metadata.get("page")
                    page_info = f", page {page + 1}" if isinstance(page, int) else ""
                    context_blocks.append(
                        f"[Source {index}: {source}{page_info}]\n{document.page_content}"
                    )
                context = "\n\n".join(context_blocks)

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a careful document QA assistant. Answer using only the retrieved context. "
                     "If the answer is not supported by the context, say you could not find it in the uploaded documents. "
                     "Do not invent details. Cite claims using source labels such as [Source 1]."),
                    ("human", "Question: {question}\n\nRetrieved context:\n{context}"),
                ])
                llm = ChatOllama(model=model_name, temperature=0)
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"question": question, "context": context})

                st.subheader("Answer")
                st.write(answer)
                with st.expander("Inspect retrieved source chunks"):
                    for index, document in enumerate(retrieved_docs, start=1):
                        source = document.metadata.get("source", "Unknown source")
                        page = document.metadata.get("page")
                        page_info = f" · Page {page + 1}" if isinstance(page, int) else ""
                        st.markdown(f"**Source {index}: {source}{page_info}**")
                        st.write(document.page_content)
                        st.divider()
            except Exception as error:
                st.error(
                    "Could not answer the question. Check that Ollama is running and the selected model is installed. "
                    f"Details: {error}"
                )

st.divider()
st.caption("Pipeline: document loaders → text splitter → embeddings → Chroma retriever → LangChain prompt/LLM chain.")
