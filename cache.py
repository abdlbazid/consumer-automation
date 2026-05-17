from functools import lru_cache
from App import call_llm_generate_code

@lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    return call_llm_generate_code(prompt)
