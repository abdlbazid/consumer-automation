import os
import boto3
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError

app = Flask(__name__)

# 1. ENFORCE HTTPS & SECURITY HEADERS (Talisman)
# Sets CSP, HSTS, and protection against XSS/Clickjacking
# In local dev, use force_https=false to avoid SSL redirect issues.
FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").strip().lower() in ("1", "true", "yes")
Talisman(app, force_https=FORCE_HTTPS)

# 2. CONFIGURE CORS (Restrict to your UI domain)
CORS(app, resources={r"/generate": {"origins": "https://your-app.com"}})

# 3. RATE LIMITING (Prevent abuse)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://" # Use Redis for multi-server production
)

# 4. INTERNAL TOKEN (for internal API calls)
# This service can run either:
#  - locally with env vars (recommended for development)
#  - in production with AWS Secrets Manager (recommended for deployment)

USE_AWS_SECRETS = os.getenv("USE_AWS_SECRETS", "false").strip().lower() in ("1", "true", "yes")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.getenv("AWS_INTERNAL_TOKEN_SECRET", "api_internal_token")


def load_production_secret():
    session = boto3.Session(region_name=AWS_REGION)
    client = session.client('secretsmanager')
    res = client.get_secret_value(SecretId=AWS_SECRET_NAME)
    return res['SecretString']

# Local / dev first (env var takes priority)
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN")

# If explicitly configured for AWS, load from Secrets Manager
if USE_AWS_SECRETS:
    try:
        INTERNAL_TOKEN = load_production_secret()
        print(f"✅ Loaded INTERNAL_TOKEN from AWS Secrets Manager ({AWS_SECRET_NAME} in {AWS_REGION})")
    except Exception as e:
        # Fail loudly in environments where AWS is expected (production)
        print(f"❌ ERROR: Could not load internal token from AWS Secrets Manager: {e}")
        raise

# For local dev, fall back to an explicit env var or a safe default (not secure)
if not INTERNAL_TOKEN:
    INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "dev-internal-token")
    print("⚠️ WARNING: Using fallback INTERNAL_TOKEN (not secure). Set INTERNAL_TOKEN env var or enable USE_AWS_SECRETS for production.")

# 5. INPUT VALIDATION SCHEMA (Marshmallow)
class GenerateRequestSchema(Schema):
    message = fields.Str(required=True, validate=lambda x: len(x) < 500)
    # marshmallow v3+ uses load_default for default values when loading input
    execute = fields.Bool(load_default=False)

@app.route('/generate', methods=['POST'])
@limiter.limit("5 per minute") # Specific limit for this expensive route
def generate():
    # Security Header Check
    if request.headers.get("X-Internal-Token") != INTERNAL_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    # Strict Schema Validation
    try:
        data = GenerateRequestSchema().load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 422

    # Logic remains UI-agnostic
    return jsonify({
        "status": "processed",
        "result": "Your backend logic here"
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Basic health / docs endpoint."""
    return jsonify({
        "status": "ok",
        "message": "This API only exposes POST /generate. Use a JSON body with {\"message\": ..., \"execute\": true/false} and set X-Internal-Token header."
    }), 200


if __name__ == '__main__':
    # In production, run this via: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run()
