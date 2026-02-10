import os
import re
import tempfile
import subprocess
import textwrap
import time
import traceback
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify
from dotenv import load_dotenv

import spacy
from openai import OpenAI
import requests
import json
import logging

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()

# ------------------- NLP -------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm") from e

# ------------------- OpenAI -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- Flask -------------------
app = Flask(__name__)

# ------------------- Config -------------------
DANGEROUS_PATTERNS = [
    r"\beval\s*\(", r"\bexec\s*\(",
]
EXECUTION_TIMEOUT = 6
MAX_USER_PROMPT_CHARS = 3000
LLM_MODEL = "gpt-5-nano"

# Import available API modules
try:
    import gmail_api
except Exception as e:
    logger.warning("gmail_api not available: %s", e)
    gmail_api = None


# ------------------- NLP Preprocessing -------------------
def preprocess_user_message(text: str) -> Dict[str, Any]:
    """Preprocess user message using spaCy NLP"""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Intent detection
    lower = text.lower()
    if any(w in lower for w in ("explain", "what is", "how to", "help me understand", "why")):
        intent = "explain"
    else:
        intent = "generate_code"

    return {
        "text": text,
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": ents,
        "noun_chunks": noun_chunks,
        "intent": intent
    }

# ------------------- Knowledge Integration -------------------
def fetch_wikipedia(query: str) -> str:
    """Fetch summary from Wikipedia"""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        response = requests.get(url, timeout=5).json()
        return response.get("extract", "")
    except Exception as e:
        return f"Wikipedia fetch error: {e}"

def fetch_internet_archive(query: str) -> str:
    """Fetch books from Internet Archive"""
    try:
        search_url = f"https://archive.org/advancedsearch.php?q={query}&fl[]=identifier&fl[]=title&output=json"
        r = requests.get(search_url, timeout=5).json()
        docs = r.get("response", {}).get("docs", [])
        snippets = []
        for doc in docs[:2]:
            identifier = doc.get("identifier")
            title = doc.get("title")
            if identifier:
                snippets.append(f"Book: {title}, Identifier: {identifier}")
        return "\n".join(snippets)
    except Exception as e:
        return f"Internet Archive fetch error: {e}"

def fetch_skillshare_britannica(query: str) -> str:
    """Placeholder for Skillshare/Britannica"""
    return f"[Skillshare/Britannica placeholder for '{query}']"

def fetch_knowledge(query: str) -> str:
    """Fetch knowledge from multiple sources"""
    wiki = fetch_wikipedia(query)
    ia = fetch_internet_archive(query)
    ss = fetch_skillshare_britannica(query)
    return "\n\n".join([wiki, ia, ss])

# ------------------- Get Available Modules -------------------
def get_available_modules_info() -> str:
    """Dynamically list all available imported modules and their functions"""
    available_modules = []
    
    # Modules to exclude (system/internal)
    exclude_modules = {'os', 're', 'tempfile', 'subprocess', 'textwrap', 'time', 'logging', 
                       'json', 'builtins', '__main__', 'sys', 'types', 'collections'}
    
    try:
        for name, obj in globals().items():
            # Skip private/dunder names and excluded modules
            if name.startswith('_') or name in exclude_modules:
                continue
            
            # Check if it's a module or has callable functions
            try:
                if hasattr(obj, '__file__') or hasattr(obj, '__dict__'):
                    funcs = []
                    for attr_name in dir(obj):
                        if not attr_name.startswith('_'):
                            try:
                                attr = getattr(obj, attr_name)
                                if callable(attr):
                                    funcs.append(attr_name)
                            except:
                                pass
                    
                    if funcs:
                        # Show first 10 functions
                        func_list = ', '.join(funcs[:10])
                        if len(funcs) > 10:
                            func_list += f", ... and {len(funcs)-10} more"
                        available_modules.append(f"  {name}: {func_list}")
            except Exception:
                pass
    
    except Exception as e:
        logger.warning("Could not list available modules: %s", e)
    
    if available_modules:
        return "\n".join(available_modules)
    else:
        return "  (Check sys.modules or imported packages for available functions)"

# ------------------- LLM Prompt Builder -------------------
def build_prompt_for_llm(nlp_summary: Dict[str, Any]) -> Tuple[str, str]:
    """Build prompt for LLM based on user intent"""
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

Generate ONLY the Python code wrapped in ```python code blocks. No explanations."""
    else:
        prompt = user_text

    return prompt, mode 

# ------------------- LLM Call -------------------
def call_llm_generate_code(prompt: str) -> str:
    """Call OpenAI LLM to generate code"""
    try:
        logger.info("LLM call: model=%s", LLM_MODEL)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate only executable Python code."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4000
            
        )
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content
            
            if content and len(content.strip()) > 0:
                return content
            else:
                return f"Empty content. Finish reason: {response.choices[0].finish_reason}"
        else:
            return "No choices in response"
            
    except Exception as e:
        logger.exception("LLM call exception: %s", e)
        return f"LLM call failed: {str(e)}"

# ------------------- Extract code -------------------
def extract_code_from_response(llm_text: str) -> Tuple[str, str]:
    """Extract Python code from LLM response"""
    # Try to find code in markdown fences
    fence_re = re.compile(r"```(?:python)?\n(.*?)```", re.S | re.I)
    match = fence_re.search(llm_text)
    
    if match:
        code = match.group(1).strip()
        explanation = fence_re.sub("", llm_text).strip()
        return code, explanation
    else:
        # Heuristic: check if response contains Python code patterns
        heur_re = re.compile(r"(def\s+\w+\(|import\s+\w+|class\s+)", re.I)
        if heur_re.search(llm_text):
            return llm_text.strip(), ""
        return "", llm_text.strip()

# ------------------- Safety Checks -------------------
def fails_safety_checks(code: str) -> Tuple[bool, str]:
    """Check if code contains dangerous patterns"""
    for patt in DANGEROUS_PATTERNS:
        if re.search(patt, code, re.I):
            return True, f"Dangerous pattern detected: {patt}"
    return False, ""

# ------------------- Run Code -------------------
def run_code_safely(code: str, timeout: int = EXECUTION_TIMEOUT) -> Dict[str, any]:
    """Execute code in a subprocess with timeout protection"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8') as tf:
        tfname = tf.name
        tf.write("# -*- coding: utf-8 -*-\n")
        tf.write("# Auto-generated script\n")
        tf.write("import sys\n")
        tf.write(f"sys.path.insert(0, r'{os.getcwd()}')\n") # Run from project directory
        tf.write(code)
        tf.flush()

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", "-u", tfname],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        runtime = time.time() - start
        result = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "runtime": runtime,
            "timeout": False
        }
    except subprocess.TimeoutExpired as ex:
        runtime = time.time() - start
        result = {
            "stdout": ex.stdout or "",
            "stderr": (ex.stderr or "") + f"\nProcess timed out after {timeout} seconds.",
            "exit_code": None,
            "runtime": runtime,
            "timeout": True
        }
    finally:
        try: 
            os.remove(tfname)
        except Exception: 
            pass

    return result

# ------------------- Flask Endpoints -------------------
@app.route("/test-llm", methods=["GET"])
def test_llm():
    """Test LLM connection"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": "Say hello in one word"}
            ],
            max_completion_tokens=100
        )
        return jsonify({
            "status": "success",
            "model": LLM_MODEL,
            "response": response.choices[0].message.content if response.choices else "No response"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """Main endpoint for code generation and execution."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Provide 'message' in JSON body"}), 400
    
    # Check max prompt length
    if len(user_message) > MAX_USER_PROMPT_CHARS:
        return jsonify({"error": f"Message exceeds {MAX_USER_PROMPT_CHARS} characters"}), 400
    
    logger.info("User request: %s", user_message)
    
    # NLP preprocessing
    nlp_summary = preprocess_user_message(user_message)

    # Build prompt for LLM
    prompt, mode = build_prompt_for_llm(nlp_summary)

    # Call LLM to generate code
    llm_text = call_llm_generate_code(prompt)

    # Process response based on mode
    if mode == "generate_code":
        # Extract code from LLM response
        code, explanation = extract_code_from_response(llm_text)
        
        # Safety checks
        dangerous, danger_msg = fails_safety_checks(code)
        safe_to_run = not dangerous

        # Execute generated code if requested and safe
        execution_result = None
        if bool(data.get("execute", True)):
            if not safe_to_run:
                execution_result = {
                    "stdout": "",
                    "stderr": danger_msg,
                    "exit_code": 1,
                    "runtime": 0,
                    "timeout": False
                }
                logger.warning("Code execution blocked: %s", danger_msg)
            elif code:
                execution_result = run_code_safely(code)
                logger.info("Code executed: exit_code=%s, runtime=%.2fs", 
                           execution_result.get("exit_code"), execution_result.get("runtime"))
    else:
        # Explanation mode
        code = ""
        explanation = llm_text
        execution_result = None
        dangerous, danger_msg = False, ""
        safe_to_run = False

    # Build response
    response = {
        "user_message": user_message,
        "mode": mode,
        "llm_raw": llm_text,
        "code": code,
        "explanation": explanation,
        "safety": {
            "dangerous": dangerous,
            "message": danger_msg,
            "safe_to_run": safe_to_run
        },
        "execution": execution_result
    }

    return jsonify(response), 200


@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoGmail AI - Code Generator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
            code { background: #f4f4f4; padding: 2px 5px; }
        </style>
    </head>
    <body>
        <h1>🚀 AutoGmail AI - Code Generator</h1>
        <p>Generate and execute Python code using OpenAI LLM with Gmail API integration.</p>
        
        <h2>API Endpoint</h2>
        <p><strong>POST</strong> <code>/generate</code></p>
        
        <h2>Request Format</h2>
        <pre>{
  "message": "your command here (e.g., 'send email to test@gmail.com saying hello')",
  "execute": true
}</pre>
        
        <h2>Examples</h2>
        <ul>
            <li>Send email: "send email to basidabd91@gmail.com saying please come to monte carlo"</li>
            <li>Write code: "write a function that counts to 10"</li>
            <li>List emails: "list all my emails"</li>
            <li>Delete email: "delete email with ID xxx"</li>
        </ul>
        
        <h2>Response Format</h2>
        <pre>{
  "user_message": "...",
  "mode": "generate_code|explain",
  "llm_raw": "raw LLM response",
  "code": "extracted Python code",
  "explanation": "explanation or full response",
  "safety": {
    "dangerous": false,
    "message": "",
    "safe_to_run": true
  },
  "execution": {
    "stdout": "output here",
    "stderr": "errors here",
    "exit_code": 0,
    "runtime": 0.5,
    "timeout": false
  }
}</pre>
        
        <h2>Quick Test</h2>
        <p><a href="/test-llm" target="_blank">Test LLM Connection</a></p>
        
        <h2>Setup Requirements</h2>
        <ul>
            <li>Python 3.8+</li>
            <li>OpenAI API key in <code>.env</code> file</li>
            <li>Gmail API credentials: <code>credentials.json</code></li>
            <li>Gmail API token: <code>token.json</code> (auto-generated on first auth)</li>
            <li>Dependencies: <code>pip install -r requirements.txt</code></li>
        </ul>
    </body>
    </html>
    """, 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "llm_model": LLM_MODEL,
        "gmail_api_available": gmail_api is not None
    }), 200

# ------------------- Entrypoint -------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    logger.info("Starting AutoGmail AI on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)