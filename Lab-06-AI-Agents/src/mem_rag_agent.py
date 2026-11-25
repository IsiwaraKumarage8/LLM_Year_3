import sqlite3
import subprocess
import time
from datetime import datetime
import pyautogui
import pygetwindow as gw
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# --- SQLite Long-Term Memory Setup ---
DB_PATH = "student_memory.db"


def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT UNIQUE
        )
    """)
    # Clean up any accidental "NONE" or empty-message entries from earlier runs
    cursor.execute("""
        DELETE FROM memories
        WHERE UPPER(TRIM(fact, '. ')) = 'NONE'
           OR fact LIKE '%don''t see any message%'
    """)
    conn.commit()
    conn.close()


def save_memory_fact(fact: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO memories (fact) VALUES (?)", (fact,))
        conn.commit()
        print(f"[SAVED TO SQLITE] {fact}")
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


# --- Connect Existing ChromaDB from Lab 5 ---
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = Chroma(
    persist_directory="../Lab-05-RAG-Pipeline/chroma_db",
    embedding_function=embeddings,
    collection_name="university_docs",
)


# --- Define Agent Tools ---
@tool
def open_calculator(expression: str) -> str:
    """Opens Windows Calculator and calculates the given mathematical expression."""
    print(f"\n[EXECUTING] Opening Windows Calculator for: {expression}")
    subprocess.Popen("calc.exe")
    time.sleep(1.5)

    try:
        calc_windows = [w for w in gw.getWindowsWithTitle("Calculator") if w.visible]
        if calc_windows:
            calc_windows[0].activate()
            time.sleep(0.3)
    except Exception:
        pass  # Windows 11 already focuses Calculator on launch

    pyautogui.write(expression, interval=0.1)
    pyautogui.press("enter")
    return f"Entered {expression} into Windows Calculator."


@tool
def get_current_datetime() -> str:
    """Retrieves the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def search_student_memory(query: str) -> str:
    """Searches stored long-term memories about the user (name, degree, preferences, facts)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT fact FROM memories")
    rows = [r[0] for r in cursor.fetchall()]
    conn.close()
    if not rows:
        return "No matching memories found."
    return "Stored user memories:\n" + "\n".join(f"- {r}" for r in rows)


@tool
def search_university_knowledge(query: str) -> str:
    """Searches the local university/NLP documents in ChromaDB for course or lecture details."""
    print(f"[EXECUTING] Searching knowledge base for: '{query}'...")
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found in the knowledge base."
    results = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        results.append(f"Source: {src}, Page: {page}\nContent: {d.page_content}")
    return "\n\n---\n\n".join(results)


# --- Initialize Agent ---
tools = [
    open_calculator,
    get_current_datetime,
    search_student_memory,
    search_university_knowledge,
]
tool_map = {t.name: t for t in tools}

llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def extract_and_store_facts(user_text: str):
    """Uses the LLM to extract personal facts to persist in SQLite."""
    extraction_prompt = (
        "Extract any personal facts about the user (such as their name, degree, interests, "
        "or preferences) from the message below as a single short sentence. "
        "If there are no personal facts to remember, reply strictly with NONE.\n\n"
        f"Message: {user_text}"
    )
    res = llm.invoke([HumanMessage(content=extraction_prompt)]).content.strip().rstrip(".")
    if res and "NONE" not in res.upper():
        print("[NEW MEMORIES DETECTED]")
        save_memory_fact(res)
    else:
        print("[MEMORY] No new information to save.")


def run_mem_rag_agent():
    init_memory_db()
    short_term_history = [
        SystemMessage(
            content=(
                "You are an Intelligent Student AI Agent. "
                "Use `search_student_memory` to recall details about the user, "
                "`search_university_knowledge` to answer questions from the ingested documents, "
                "`get_current_datetime` for the current date or time, "
                "and `open_calculator` for any math calculations."
            )
        )
    ]

    print("=" * 45)
    print("INTELLIGENT STUDENT AI AGENT (MEMORY + RAG)")
    print("=" * 45)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        # 1. Update Long-Term SQLite Memory
        extract_and_store_facts(user_input)

        # 2. Update Short-Term Working Memory
        short_term_history.append(HumanMessage(content=user_input))

        # 3. Invoke LLM with Tools
        response = llm_with_tools.invoke(short_term_history)
        short_term_history.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                print(f"[TOOL SELECTED] {name}")
                print(f"[TOOL ARGUMENTS] {args}")

                tool_output = tool_map[name].invoke(args)
                print(f"[TOOL RESULT] {tool_output}")

                short_term_history.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

            final_reply = llm_with_tools.invoke(short_term_history)
            short_term_history.append(final_reply)
            print(f"\nAgent: {final_reply.content}")
        else:
            print(f"\nAgent: {response.content}")


if __name__ == "__main__":
    run_mem_rag_agent()