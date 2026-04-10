from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import requests
from flask_cors import CORS

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# --- NEW: IMPORT FROM ASYNCCALL.PY ---
from asynccall import execute_backend_async

# Initialize Sentry only when a DSN is provided (avoid errors in local dev)
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FlaskIntegration()],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        send_default_pii=True,
    )
else:
    print("⚠️ Sentry disabled: set SENTRY_DSN to enable error reporting")

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "*"}})

AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:5000/generate")

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    execute_flag = bool(data.get("execute", False))

    if len(message) > 5000:
        return jsonify({"error": "Message too long (max 5000 caracters)"})
    
    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    if not AI_BACKEND_URL.startswith("http"):
         return jsonify({"error": "Invalid AI_BACKEND_URL"}), 500 
    
    # -----------------------------
    # CONNECT TO REAL BACKEND (via imported asynccall.py logic)
    try:
        # Start the async thread
        internal_token = os.getenv("INTERNAL_API_TOKEN")
        thread, result_dict = execute_backend_async(message, execute_flag, AI_BACKEND_URL, internal_token)
        
        # Wait for the thread to finish (as in your original logic)
        thread.join(timeout=35)
        
        backend_data = result_dict.get("response", {})

        if "error" in backend_data:
            raise Exception(backend_data["error"])

        code = backend_data.get("code", "")
        explanation = backend_data.get("explanation", "")
        execution_result = backend_data.get("execution")

        # Backend already handled execution
        execute_flag = False
 
    except Exception as e:
        sentry_sdk.capture_exception(e)
        code = ""
        explanation = "Failed to connect to AI backend"
        execution_result = {
            "stdout": "",
            "stderr": str(e),
            "exit_code": None
        }
    # -----------------------------

    # Fallback local execution (only if backend did NOT run it)
    if execute_flag and code:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tfname = tf.name
            tf.write(code)
            tf.flush()

        try:
            proc = subprocess.run(
                ["python", tfname],
                capture_output=True,
                text=True,
                timeout=5
            )
            execution_result = {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode
            }
        except subprocess.TimeoutExpired:
            execution_result = {
                "stdout": "",
                "stderr": "Timeout",
                "exit_code": None
            }
        finally:
            try:
                os.remove(tfname)
            except Exception:
                pass

    response = {
        "user_message": message,
        "code": code,
        "explanation": explanation,
        "execution": execution_result
    }

    return jsonify(response)


@app.route("/", methods=["GET"])
def index():
    return "Enabler AI Connector Backend Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
