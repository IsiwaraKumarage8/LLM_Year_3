import os
import time
import glob
import platform
import subprocess
from datetime import datetime
import psutil
import pyautogui
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
    return f"Entered '{expression}' into Windows Calculator."


@tool
def get_current_datetime() -> str:
    """Retrieves the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def write_in_notepad(text: str) -> str:
    """Opens Windows Notepad and types the provided text."""
    print(f"\n[EXECUTING] Opening Notepad and writing text...")
    subprocess.Popen("notepad.exe")
    time.sleep(1.5)

    try:
        notepad_windows = [w for w in gw.getWindowsWithTitle("Notepad") if w.visible]
        if notepad_windows:
            notepad_windows[0].activate()
            time.sleep(0.3)
    except Exception:
        pass

    pyautogui.write(text, interval=0.05)
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
    """Creates a text file at the specified filepath with the given content."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
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

llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# 3. Agent Execution Loop
def run_agent():
    system_prompt = SystemMessage(
        content=(
            "You are a helpful Windows assistant with access to local system tools. "
            "Always use the appropriate tool when asked to calculate, check date/time, "
            "open Notepad, search files, create files, or inspect system specs. "
            "Never perform math yourself—always call open_calculator."
        )
    )
    print("Multi-Tool Agent Ready! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [system_prompt, HumanMessage(content=user_input)]
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                print(f"[TOOL SELECTED] {name}")
                print(f"[TOOL ARGUMENTS] {args}")

                result = tool_map[name].invoke(args)
                print(f"[TOOL RESULT] {result}\n")

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            final_response = llm_with_tools.invoke(messages)
            print(f"Agent: {final_response.content}\n")
        else:
            print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    run_agent()