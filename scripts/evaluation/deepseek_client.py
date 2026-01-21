# scripts/evaluation/deepseek_client.py
import os
import time
import requests

class DeepSeekClient:
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout: int = 120,
        retries: int = 5,
        backoff_base: float = 1.8,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.backoff_base = backoff_base

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env}. e.g. export {api_key_env}='...'")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        last_err = None
        for attempt in range(self.retries):
            try:
                r = requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                return j["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(self.backoff_base ** attempt)

        raise RuntimeError(f"DeepSeek call failed after {self.retries} retries: {last_err}")
