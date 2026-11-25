import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "./chroma_db"
COLLECTION_NAME = "university_docs"

st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")
st.title("🎓 University AI Assistant (Query Expansion RAG)")
st.caption("Enhanced with multi-query retrieval")


@st.cache_resource
def load_rag_components():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return embeddings, llm, vector_store


embeddings, llm, vector_store = load_rag_components()

# Document Upload Sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload university PDFs to add to knowledge base",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process & Ingest PDFs"):
        with st.spinner("Processing PDF(s)..."):
            new_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyMuPDFLoader(tmp_path)
                new_docs.extend(loader.load())
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(new_docs)
            vector_store.add_documents(chunks)
            st.success(f"Added {len(chunks)} chunks to vector store!")

# Chat state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_input = st.chat_input("Ask a university question...")

if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Expanding query and searching documents..."):
            # 1. Query Expansion Chain
            expansion_prompt = ChatPromptTemplate.from_template(
                """Generate 3 alternate ways to ask the following question to retrieve relevant information from a university document database:
Question: {question}"""
            )
            query_generator = expansion_prompt | llm | StrOutputParser()
            expanded_raw = query_generator.invoke({"question": prompt_input})

            queries = [q.strip() for q in expanded_raw.strip().split("\n") if q.strip()]
            queries.append(prompt_input)

            # 2. Multi-Query Retrieval & Deduplication
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            retrieved_docs_dict = {}
            for q in queries:
                for doc in retriever.invoke(q):
                    retrieved_docs_dict[doc.page_content] = doc

            unique_docs = list(retrieved_docs_dict.values())
            context_text = "\n\n".join([doc.page_content for doc in unique_docs])

            # 3. Final Response Generation
            qa_prompt = ChatPromptTemplate.from_template(
                """Answer the question based ONLY on the context provided below. If not found, state that it is unavailable.

Context:
{context}

Question: {question}

Answer:"""
            )
            formatted_prompt = qa_prompt.format(context=context_text, question=prompt_input)
            response = llm.invoke(formatted_prompt)
            st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})