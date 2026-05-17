"""Shared core logic for the Enabler AI backend.

This module contains the core LLM + execution logic used by both
`app.py` (main server) and `api.py` (internal API bridge).

It is intentionally self-contained so that both services can import it
without creating circular dependencies.
"""

import os
import re
import tempfile
import subprocess
import time
import logging
from typing import Dict, Any, Tuple

import requests
from openai import OpenAI

from awsstore import get_keys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DANGEROUS_PATTERNS = [r"\beval\s*\(", r"\bexec\s*\("]
EXECUTION_TIMEOUT = 60
MAX_USER_PROMPT_CHARS = 3000
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")

# ---------------------------------------------------------------------------
# OpenAI client initialization (uses awsstore.get_keys for local/AWS fallback)
# ---------------------------------------------------------------------------
keys = get_keys()
OPENAI_API_KEY = keys.get("openai_key")
GMAIL_TOKEN = keys.get("gmail_token")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. LLM calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

def preprocess_user_message(text: str) -> Dict[str, Any]:
    """Preprocess user message for later prompting."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("Run: python -m spacy download en_core_web_sm") from e

    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    lower = text.lower()
    intent = "explain" if any(w in lower for w in ("explain", "what is", "how to", "help me understand", "why")) else "generate_code"

    return {
        "text": text,
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": ents,
        "noun_chunks": noun_chunks,
        "intent": intent,
    }

# ---------------------------------------------------------------------------
# Prompting / model call
# ---------------------------------------------------------------------------

def build_prompt_for_llm(nlp_summary: Dict[str, Any]) -> Tuple[str, str]:
    """Build a prompt for the LLM based on intent."""
    user_text = nlp_summary["text"]
    mode = nlp_summary.get("intent", "chat")

    if mode == "generate_code":
        prompt = f"""You are a Python code generator. Generate ONLY executable Python code to accomplish the task below.

You can import and use the gmail_api module for email tasks. Available functions:
- send_email(to, subject, body, attachments=None) - Send an email immediately
- create_draft(to, subject, body, attachments=None) - Create an email draft
- send_draft(draft_id) - Send an existing draft
- list_messages(query=None, label_ids=None, max_results=50) - List messages
- get_message(message_id, format='full') - Get a specific message
- delete_message(message_id) - Delete a message
- modify_labels(message_id, add_labels=None, remove_labels=None) - Modify labels
- list_labels() - List all labels
- get_attachment(message_id, attachment_id, save_path) - Download attachment

Task: {user_text}

Generate ONLY the Python code wrapped in ```python code blocks. No explanations.

dont do anything thats dangeous or against te law always comply with open ai usage policies
"""
    else:
        prompt = user_text

    return prompt, mode


def call_llm_generate_code(prompt: str) -> str:
    """Call the OpenAI LLM and return the raw text response."""
    if not client:
        raise RuntimeError("OpenAI client not initialized (missing OPENAI_API_KEY)")

    try:
        logger.info("LLM call: model=%s", LLM_MODEL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate only executable Python code."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4000,
        )

        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content
            if content and len(content.strip()) > 0:
                return content
            return f"Empty content. Finish reason: {response.choices[0].finish_reason}"
        return "No choices in response"

    except Exception as e:
        logger.exception("LLM call exception: %s", e)
        return f"LLM call failed: {str(e)}"


# ---------------------------------------------------------------------------
# Code extraction + safety + execution
# ---------------------------------------------------------------------------

def extract_code_from_response(llm_text: str) -> Tuple[str, str]:
    """Extract Python code from the LLM response."""
    fence_re = re.compile(r"```(?:python)?\n(.*?)```", re.S | re.I)
    match = fence_re.search(llm_text)

    if match:
        code = match.group(1).strip()
        explanation = fence_re.sub("", llm_text).strip()
        return code, explanation

    heur_re = re.compile(r"(def\s+\w+\(|import\s+\w+|class\s+)", re.I)
    if heur_re.search(llm_text):
        return llm_text.strip(), ""

    return "", llm_text.strip()


def fails_safety_checks(code: str) -> Tuple[bool, str]:
    for patt in DANGEROUS_PATTERNS:
        if re.search(patt, code, re.I):
            return True, f"Dangerous pattern detected: {patt}"
    return False, ""


def run_code_safely(code: str, timeout: int = EXECUTION_TIMEOUT) -> Dict[str, Any]:
    """Execute code in a subprocess with timeout protection."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tfname = tf.name
        tf.write("# -*- coding: utf-8 -*-\n")
        tf.write("# Auto-generated script\n")
        tf.write("import sys\n")
        tf.write(f"sys.path.insert(0, r'{os.getcwd()}')\n")
        tf.write(code)
        tf.flush()

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", "-u", tfname],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        runtime = time.time() - start
        result = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "runtime": runtime,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as ex:
        runtime = time.time() - start
        result = {
            "stdout": ex.stdout or "",
            "stderr": (ex.stderr or "") + f"\nProcess timed out after {timeout} seconds.",
            "exit_code": None,
            "runtime": runtime,
            "timeout": True,
        }
    finally:
        try:
            os.remove(tfname)
        except Exception:
            pass

    return result


def generate_response(user_message: str, execute: bool = True) -> Dict[str, Any]:
    """Generate a structured response for the user request."""
    nlp_summary = preprocess_user_message(user_message)
    prompt, mode = build_prompt_for_llm(nlp_summary)
    llm_text = call_llm_generate_code(prompt)

    if mode == "generate_code":
        code, explanation = extract_code_from_response(llm_text)
        dangerous, danger_msg = fails_safety_checks(code)
        safe_to_run = not dangerous

        execution_result = None
        if execute:
            if not safe_to_run:
                execution_result = {
                    "stdout": "",
                    "stderr": danger_msg,
                    "exit_code": 1,
                    "runtime": 0,
                    "timeout": False,
                }
            elif code:
                execution_result = run_code_safely(code)

    else:
        code = ""
        explanation = llm_text
        execution_result = None
        dangerous, danger_msg = False, ""
        safe_to_run = False

    return {
        "user_message": user_message,
        "mode": mode,
        "llm_raw": llm_text,
        "code": code,
        "explanation": explanation,
        "safety": {
            "dangerous": dangerous,
            "message": danger_msg,
            "safe_to_run": safe_to_run,
        },
        "execution": execution_result,
    }
