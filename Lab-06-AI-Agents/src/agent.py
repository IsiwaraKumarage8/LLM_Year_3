import os
import time
import glob
import platform
import subprocess
from datetime import datetime
import psutil
import pyautogui
import pyperclip
import pygetwindow as gw
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# 1. Define Tools
@tool
def open_calculator(expression: str) -> str:
    """Opens Windows Calculator and enters a mathematical expression (e.g., '7*4' or '25*4')."""
    print(f"\n[EXECUTING] Opening Windows Calculator for: {expression}")
    subprocess.Popen("calc.exe")
    time.sleep(1.0)

    try:
        calc_windows = [w for w in gw.getWindowsWithTitle("Calculator") if w.visible]
        if calc_windows:
            calc_windows[0].activate()
            time.sleep(0.2)
    except Exception:
        pass

    pyautogui.write(expression, interval=0.05)
    pyautogui.press("enter")
    return f"Entered '{expression}' into Windows Calculator."


@tool
def get_current_datetime() -> str:
    """Retrieves the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def write_in_notepad(text: str) -> str:
    """Opens Windows Notepad and pastes the provided plain-text string."""
    # Fallback if a small LLM passes the tool name instead of the actual date string
    if text.strip() in {"get_current_datetime", "get_current_datetime()"}:
        text = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")

    print("\n[EXECUTING] Opening Notepad and pasting text...")
    subprocess.Popen("notepad.exe")
    time.sleep(1.0)

    try:
        notepad_windows = [w for w in gw.getWindowsWithTitle("Notepad") if w.visible]
        if notepad_windows:
            notepad_windows[0].activate()
            time.sleep(0.2)
    except Exception:
        pass

    pyperclip.copy(str(text))
    pyautogui.hotkey("ctrl", "v")
    return f"Opened Notepad and wrote: '{text}'"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Searches for files by name or extension (e.g., '*.pdf') in a specified directory."""
    search_path = os.path.join(directory, "**", pattern)
    matches = glob.glob(search_path, recursive=True)
    if not matches:
        return f"No files matching '{pattern}' found in '{directory}'."
    return "Found files:\n" + "\n".join(matches)


@tool
def create_text_file(filepath: str, content: str) -> str:
    """Creates a text file at the specified filepath with the given plain-text content."""
    # Fallback if a small LLM passes the tool name instead of the actual date string
    if content.strip() in {"get_current_datetime", "get_current_datetime()"}:
        content = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(str(content))
    return f"Created file '{filepath}' with content: '{content}'"


@tool
def get_system_info() -> str:
    """Retrieves operating system, processor, and memory specifications."""
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return (
        f"OS: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"Total RAM: {ram_gb} GB"
    )


# 2. Initialize LLM & Bind Tools
tools = [
    open_calculator,
    get_current_datetime,
    write_in_notepad,
    search_files,
    create_text_file,
    get_system_info,
]
tool_map = {t.name: t for t in tools}

TERMINAL_TOOLS = {"write_in_notepad", "create_text_file", "open_calculator"}

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=2048,
    num_predict=150,
    keep_alive="10m",
)
llm_with_tools = llm.bind_tools(tools)


# 3. Agent Execution Loop with Sequential Tool Chaining
def run_agent():
    system_prompt = SystemMessage(
        content=(
            "You are a Windows assistant with local tools. "
            "Call ONLY ONE tool per step and wait for its result. "
            "When passing a value from a previous tool (like `get_current_datetime`) to "
            "`create_text_file` or `write_in_notepad`, pass the ACTUAL date/time value returned "
            "(e.g., '2026-11-14 14:07:04 (Saturday)'), NEVER the tool name string."
        )
    )
    print("Fast Multi-Tool Chaining Agent Ready! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [system_prompt, HumanMessage(content=user_input)]
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        step = 1
        final_printed = False

        while response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                print(f"[STEP {step} - TOOL SELECTED] {name}")
                print(f"[STEP {step} - TOOL ARGUMENTS] {args}")

                try:
                    result = tool_map[name].invoke(args)
                    succeeded = True
                except Exception as e:
                    result = f"Error: {e}."
                    succeeded = False

                print(f"[STEP {step} - TOOL RESULT] {result}\n")
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                step += 1

                if succeeded and name in TERMINAL_TOOLS:
                    print(f"Agent: Done! {result}\n")
                    final_printed = True

            if final_printed:
                break

            response = llm_with_tools.invoke(messages)
            messages.append(response)

        if not final_printed:
            print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    run_agent()