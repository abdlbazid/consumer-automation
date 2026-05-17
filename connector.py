import os
import logging
import tempfile
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

# AWS & Monitoring Imports
# import sentry_sdk
# from sentry_sdk.integrations.flask import FlaskIntegration
from awsstore import get_keys  # Your existing AWS loader
from asynccall import execute_backend_async

# 1. SETUP LOGGING (To see what's happening in the terminal)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 2. LOAD EVERYTHING FROM AWS SECRETS MANAGER
# This removes the need for any .env or os.getenv calls.
try:
    aws_config = get_keys()
    INTERNAL_TOKEN = aws_config.get("INTERNAL_API_TOKEN")
    SENTRY_DSN = aws_config.get("SENTRY_DSN")
    logger.info("✅ Successfully loaded configuration from AWS Secrets.")
except Exception as e:
    logger.error(f"❌ Failed to load AWS Secrets: {e}")
    aws_config = {}
    INTERNAL_TOKEN = None
    SENTRY_DSN = None

# 3. SET THE HARDCODED PRODUCTION LINK (Your Fixed IP)
# This is the "Bridge" to your AI Backend server.
AI_BACKEND_URL = "http://3.80.95.49:5000"

# 4. INITIALIZE SENTRY (Error Tracking)
# if SENTRY_DSN:
#     sentry_sdk.init(
#         dsn=SENTRY_DSN,
#         integrations=[FlaskIntegration()],
#         traces_sample_rate=1.0
#     )

# ---------------------------------------------------------
# MAIN LOGIC: THE GENERATE ENDPOINT
# ---------------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    Takes the user message, sends it to the AI Backend at 3.80.95.49,
    and returns the result. Includes a 15-second safety timeout.
    """
    # Security: Check the Token from AWS
    # token = request.headers.get("X-Internal-Token")
    # if not INTERNAL_TOKEN or token != INTERNAL_TOKEN:
    #     logger.warning("Blocked an unauthorized request attempt.")
    #     return jsonify({"error": "Unauthorized"}), 401

    # Parse User Data
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    execute_flag = bool(data.get("execute", False))

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        logger.info(f"Connecting to AI Backend at {AI_BACKEND_URL}...")
        
        # Call the Backend using your internal async logic
        thread, result_dict = execute_backend_async(
            message, 
            execute_flag, 
            AI_BACKEND_URL, 
            INTERNAL_TOKEN
        )
        
        # THE KILL SWITCH: If the server at .49 doesn't answer in 15s, stop.
        # This prevents the 3-minute hang.
        thread.join(timeout=15)
        
        if thread.is_alive():
            raise Exception("AI Backend at 3.80.95.49 timed out (15s).")

        backend_data = result_dict.get("response", {})
        
        if "error" in backend_data:
            raise Exception(backend_data["error"])

        # Success! Send the AI response to the user
        return jsonify(backend_data), 200

    except Exception as e:
        logger.error(f"Connection Error: {str(e)}")
        # sentry_sdk.capture_exception(e)
        
        # Detailed error response for debugging
        return jsonify({
            "user_message": message,
            "explanation": f"Could not reach AI Backend: {str(e)}",
            "execution": {
                "stdout": "",
                "stderr": "Check if Port 8000 is open in AWS Security Groups.",
                "exit_code": 1
            }
        }), 500

# ---------------------------------------------------------
# SYSTEM STATUS CHECK
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return f"🚀 AutoGmail AI Connector is LIVE. Target: {AI_BACKEND_URL}"

if __name__ == "__main__":
    # AWS typically uses port 8000 for Gunicorn/Connector
    app.run(host="0.0.0.0", port=8000, debug=False)