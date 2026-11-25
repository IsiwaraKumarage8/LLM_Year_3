import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Define paths and collection name
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "university_docs"


def main():
    print("Loading PDF documents from directory...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    if not documents:
        print(f"No PDF files found in '{DOCS_DIR}'. Please place university PDFs in that folder.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")

    # 2. Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Initialize Ollama Embedding model
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # 4. Save embeddings into ChromaDB vector store
    print("Storing chunks in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Ingestion complete! ChromaDB is ready for retrieval.")


if __name__ == "__main__":
    main()