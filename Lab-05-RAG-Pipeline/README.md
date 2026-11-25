# Lab 05: University Knowledge Assistant — Query Expansion RAG Pipeline

**Module:** PUSL 3205 — Large Language Models and Generative AI  
**Lab Session:** Lab 05 — Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB & Ollama  
**Author:** Kumarage Isiwara Didulani (`IsiwaraKumarage8`)

---

## 1. Overview
This directory contains a complete **Retrieval-Augmented Generation (RAG)** pipeline designed to answer academic questions grounded strictly in university handbooks and lecture PDFs. Instead of relying solely on single-query naive vector search, the assistant implements **Multi-Query Expansion** alongside an interactive **Streamlit Web Interface** that supports live sidebar PDF ingestion at runtime.

All language generation and dense embedding computations execute **100% locally** via **Ollama** and **ChromaDB**, ensuring zero external API costs and full local data privacy.

---

## 2. Repository Structure
```text
Lab-05-RAG-Pipeline/
├── README.md
├── requirements.txt
├── docs/                                   # University handbook & lecture PDFs (excluded from Git via .gitignore)
├── src/
│   ├── ingest.py                           # Loads PDFs via PyMuPDF, chunks text, and indexes embeddings in ChromaDB
│   ├── rag_pipeline.py                     # CLI terminal Q&A assistant with multi-query expansion
│   └── app.py                              # Streamlit web UI with sidebar PDF upload & multi-query RAG chat
└── screenshots/
    ├── 01_ollama_and_ingestion.png         # Local Ollama models & ChromaDB chunk ingestion log
    ├── 02_cli_rag_pipeline.png             # CLI RAG evaluation with query expansion & abstention grounding
    └── 03_streamlit_rag_ui.png             # Streamlit University AI Assistant web interface

3. System Architecture & Pipeline Workflow

[University PDFs] ──► PyMuPDFLoader ──► RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
                                                      │
                                                      ▼
                                        qwen3-embedding:0.6b (Ollama)
                                                      │
                                                      ▼
                                       Local ChromaDB ("university_docs")
                                                      ▲
                                                      │ (Top-k Deduplicated Retrieval)
[User Question] ──► llama3.2:3b (Query Expansion: 3+ query variations) ──► Grounded Prompt ──► Final Answer

### Technical Specifications

* **Generator LLM:** `llama3.2:3b` (via Ollama) — `temperature = 0.2` for deterministic, context-grounded answers
* **Embedding Model:** `qwen3-embedding:0.6b` (via Ollama) — Dense semantic embeddings (`639 MB` local model)
* **Vector Database:** `ChromaDB` (`langchain-chroma`) — Persistent local store (`./chroma_db`, collection: `university_docs`)
* **Document Loader:** `PyMuPDFLoader` (`pymupdf`) — Extracts text page-by-page from multi-page university handbooks
* **Text Splitter:** `RecursiveCharacterTextSplitter` — `chunk_size = 1000`, `chunk_overlap = 200` to preserve section boundaries
* **Retrieval Strategy:** Multi-Query Expansion + Deduplication — Generates alternate query phrasings, retrieves top-k chunks per variation, and deduplicates exact matches
* **User Interfaces:** CLI (`rag_pipeline.py`) & Web UI (`app.py`) — Terminal Q&A loop + Streamlit app with sidebar multi-PDF upload (`st.file_uploader`)