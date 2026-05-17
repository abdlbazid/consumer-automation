import os
import threading
import requests


def execute_backend_async(message: str, execute_flag: bool, backend_url: str, internal_token: str | None = None):
    """Call the AI backend in a background thread.

    Returns (thread, result_dict). The caller can join() the thread or poll result_dict.
    """

    result = {"response": None}

    def _worker():
        try:
            headers = {"Content-Type": "application/json"}
            if internal_token:
                headers["X-Internal-Token"] = internal_token

            resp = requests.post(
                backend_url,
                headers=headers,
                json={"message": message, "execute": execute_flag},
                timeout=30,
            )
            result["response"] = resp.json()
        except Exception as e:
            result["response"] = {"error": str(e)}

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, result
