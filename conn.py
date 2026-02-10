from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# URL of the real AI backesnd (app.py)
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:5000/generate")


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    execute_flag = bool(data.get("execute", False))

    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    # -----------------------------
    # CONNECT TO REAL BACKEND (app.py)
    try:
        backend_response = requests.post(
            AI_BACKEND_URL,
            json={
                "message": message,
                "execute": execute_flag
            },
            timeout=30
        )
        backend_response.raise_for_status()
        backend_data = backend_response.json()

        code = backend_data.get("code", "")
        explanation = backend_data.get("explanation", "")
        execution_result = backend_data.get("execution")

        # Backend already handled execution
        execute_flag = False

    except Exception as e:
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
    # IMPORTANT: use a different port than app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
