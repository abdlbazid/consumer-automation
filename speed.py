import requests

from conn import AI_BACKEND_URL


backend_response = requests.post(
    AI_BACKEND_URL,
    json={"message": message,"execute": execute_flag},
    timeout=30
)
