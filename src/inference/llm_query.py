"""
Sends the serialized diagram graph + a user question to an LLM - either a
local Ollama model, or a hosted model via OpenRouter.

--- Ollama (local, free, needs your own hardware) ---
Requires Ollama running locally (`ollama serve`, default port 11434) and a
model already pulled, e.g.:

    ollama pull llama3.1

--- OpenRouter (hosted, paid per token, no local hardware needed) ---
1. Sign up at https://openrouter.ai and add credit to your account.
2. Get an API key from https://openrouter.ai/keys
3. Set it as an environment variable rather than hardcoding it in source
   (this file may end up in git/shared with others):

    Windows (cmd):        set OPENROUTER_API_KEY=sk-or-...
    Windows (powershell):  $env:OPENROUTER_API_KEY="sk-or-..."
    (set it before running `python app.py`, in the same terminal session -
    or add it to your system's persistent environment variables)

4. Pick a model slug from https://openrouter.ai/models - e.g.:
    "meta-llama/llama-3.1-70b-instruct"
    "anthropic/claude-3.5-haiku"
    "openai/gpt-4o-mini"

Install the one extra dependency if you don't already have it:

    pip install requests
"""

import os
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT_TEMPLATE = """You are a P&ID (Piping & Instrumentation Diagram) assistant.

You are given a structured extraction of a diagram: equipment/instrument tags,
their types, and how they are connected to each other. Answer the user's
question using ONLY this data.

Important: this data comes from an automated computer-vision detection
pipeline and can contain errors - missed symbols, misread tags, or missed
connections. If a question hinges on a detail that looks incomplete,
inconsistent, or absent from the data, say so explicitly rather than
guessing or inventing an answer.

DIAGRAM DATA:
{graph_text}
"""


def _build_messages(question, graph_text, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(graph_text=graph_text)}]
    if history:
        for m in history:
            role = m.get("role")
            content = m.get("content")
            # Defensive: some Gradio versions attach extra structure to
            # stored chat messages rather than a plain string; both Ollama
            # and OpenRouter reject non-string content.
            if content is not None and not isinstance(content, str):
                content = str(content)
            if role and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def ask_diagram_question(question, graph_text, model="llama3.1", history=None, timeout=120):
    """Local Ollama backend. See module docstring for setup."""
    messages = _build_messages(question, graph_text, history)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return ("Could not reach Ollama at localhost:11434. Make sure it's "
                "running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {model}`).")
    except requests.exceptions.Timeout:
        return "Ollama took too long to respond (timed out). Try a smaller model or a shorter question."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text
        except Exception:
            pass
        print(f"DEBUG: Ollama HTTP {resp.status_code} - request messages were:")
        for m in messages:
            print(f"  role={m.get('role')!r} content_type={type(m.get('content')).__name__} "
                  f"preview={str(m.get('content'))[:80]!r}")
        print(f"DEBUG: Ollama response body: {body[:1000]}")
        return (f"Ollama returned an error (HTTP {resp.status_code}): {body[:300] or e}. "
                f"If this mentions the model name, run `ollama pull {model}` first. "
                f"Full request/response details printed to the terminal.")
    except Exception as e:
        return f"Unexpected error talking to Ollama: {e}"

    try:
        return resp.json()["message"]["content"]
    except (KeyError, ValueError) as e:
        return f"Got a response from Ollama but couldn't parse it ({e}). Raw response: {resp.text[:500]}"


def ask_diagram_question_openrouter(question, graph_text, model="meta-llama/llama-3.1-70b-instruct",
                                     history=None, timeout=120, api_key=None):
    """
    Hosted backend via OpenRouter. Costs real money per call - see module
    docstring for setup. api_key defaults to the OPENROUTER_API_KEY
    environment variable if not passed explicitly.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ("No OpenRouter API key found. Set the OPENROUTER_API_KEY environment "
                "variable (see the top of llm_query.py for how) or pass api_key= directly.")

    messages = _build_messages(question, graph_text, history)

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                # Optional but recommended by OpenRouter for their own analytics/rankings -
                # harmless to leave as-is, change if you want your own app name to show.
                "HTTP-Referer": "http://localhost",
                "X-Title": "PID Symbol Detector",
            },
            json={"model": model, "messages": messages},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "Could not reach OpenRouter - check your internet connection."
    except requests.exceptions.Timeout:
        return "OpenRouter took too long to respond (timed out). Try a shorter question."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text
        except Exception:
            pass
        print(f"DEBUG: OpenRouter HTTP {resp.status_code} - response body: {body[:1000]}")
        hint = ""
        if resp.status_code == 401:
            hint = " Check that OPENROUTER_API_KEY is set correctly."
        elif resp.status_code == 402:
            hint = " Your OpenRouter account is out of credit."
        elif resp.status_code == 404:
            hint = f" Check that '{model}' is a valid model slug on openrouter.ai/models."
        return f"OpenRouter returned an error (HTTP {resp.status_code}): {body[:300] or e}.{hint}"
    except Exception as e:
        return f"Unexpected error talking to OpenRouter: {e}"

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        return f"Got a response from OpenRouter but couldn't parse it ({e}). Raw response: {resp.text[:500]}"


if __name__ == "__main__":
    # Quick manual test - point this at a page you've already processed.
    from src.inference.pipeline import process_full_page
    from src.inference.diagram_graph import build_diagram_graph, graph_to_text

    page_data = process_full_page("pdf_pages/page_1.png")
    graph = build_diagram_graph(page_data, page_label="Page 1")
    graph_text = graph_to_text(graph)

    print("--- Graph sent to LLM ---")
    print(graph_text)
    print("--- Asking a test question (Ollama) ---")
    answer = ask_diagram_question("What equipment is connected to the first pump you find?", graph_text)
    print(answer)

    # Uncomment to test OpenRouter instead (requires OPENROUTER_API_KEY set):
    # print("--- Asking a test question (OpenRouter) ---")
    # answer = ask_diagram_question_openrouter("How many ball valves are there?", graph_text)
    # print(answer)
