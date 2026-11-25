# Lab 06: Tool-Using AI Agents — Windows Automation, SQLite Memory & RAG Integration

**Module:** PUSL 3205 — Large Language Models and Generative AI  
**Lab Session:** Lab 06 — Autonomous Tool-Calling Agents with LangChain, Ollama, SQLite & ChromaDB  
**Author:** Kumarage Isiwara Didulani (`IsiwaraKumarage8`)

---

## 1. Overview
This directory implements two local, tool-calling AI agents powered by **Ollama (`llama3.2:3b`)** and **LangChain** in **PUSL 3205**:

1. **Multi-Tool Windows Automation Agent (`src/agent.py` — Part A & Exercise):** Executes sequential tool chaining across desktop GUI applications (`calc.exe`, `notepad.exe`), local file operations, system diagnostics, and real-time clock queries.
2. **Memory- & RAG-Powered Student AI Agent (`src/mem_rag_agent.py` — Parts B & C):** Combines **short-term working memory** (conversation history), **persistent long-term SQLite memory** (`student_memory.db`), and **ChromaDB vector retrieval** (`university_docs` from Lab 05) to deliver personalized, document-grounded student assistance.

---

## 2. Repository Structure
```text
Lab-06-AI-Agents/
├── README.md
├── requirements.txt
├── src/
│   ├── agent.py                        # Multi-tool Windows automation & sequential tool chaining agent
│   └── mem_rag_agent.py                # Dual-memory (working + SQLite) & ChromaDB RAG-powered student agent
└── screenshots/
    ├── 01_multi_tool_agent.png         # Tool selection & Windows Calculator / Notepad / System Info execution
    └── 02_memory_rag_agent.png         # SQLite long-term memory persistence & ChromaDB handbook retrieval