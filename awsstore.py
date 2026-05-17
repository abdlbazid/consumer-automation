import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def _load_local_keys():
    """Try to load secrets from environment variables or local files.

    This allows the project to run locally without requiring AWS credentials.
    """
    # Prefer explicit env vars.
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    gmail_token = os.getenv("GMAIL_TOKEN")
    if openai_key or gmail_token:
        return {"openai_key": openai_key, "gmail_token": gmail_token}

    # Fallback: token.json (OAuth token saved by gmail_api setup)
    try:
        with open("token.json", "r", encoding="utf-8") as f:
            token_data = json.load(f)
            gmail_token = token_data.get("token") or token_data.get("access_token")
            if gmail_token:
                return {"openai_key": openai_key, "gmail_token": gmail_token}
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"❌ Error reading token.json: {e}")

    return None


def get_keys(secret_name="AutoGmailKeys"):
    # First try local sources so the project can run without AWS credentials.
    local = _load_local_keys()
    if local:
        print("✅ Loaded keys from environment or token.json")
        return local

    # Fall back to AWS Secrets Manager.
    # It will attempt to read credentials from ~/.aws/credentials or env vars.
    session = boto3.Session(region_name="eu-north-1")
    client = session.client("secretsmanager")

    # This is the "Full Path" (ARN) to your secret
    secret_id = "arn:aws:secretsmanager:eu-north-1:115417277601:secret:AutoGmailKeys"

    try:
        response = client.get_secret_value(SecretId=secret_id)
        secret_dict = json.loads(response["SecretString"])
        print("✅ Success: Credentials picked up from AWS Secrets Manager")
        return secret_dict

    except NoCredentialsError:
        print("❌ Error: AWS credentials not found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or run 'aws configure'.")
        return {}
    except ClientError as e:
        print(f"❌ Error: {e.response['Error']['Message']}")
        return {}


# --- EXECUTE ---
keys = get_keys()

gmail_token = keys.get("gmail_token")
openai_key = keys.get("openai_key")
print(f"Gmail Token Loaded: {bool(gmail_token)}")
print(f"OpenAI Key Loaded: {bool(openai_key)}")
