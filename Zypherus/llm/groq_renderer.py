"""Groq LLM renderer - uses Groq API instead of Ollama."""

import logging
import os
from typing import Iterator, Optional

logger = logging.getLogger("ZYPHERUS.LLM.GROQ")


class GroqRenderer:
    """Groq API renderer for streaming responses."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Token limits for different operations
        self.tokens_answer = int(os.getenv("ACE_TOKENS_ANSWER", "256"))
        self.tokens_verify = int(os.getenv("ACE_TOKENS_VERIFY", "200"))
        self.tokens_default = int(os.getenv("ACE_TOKENS_DEFAULT", "200"))
        
        logger.info(f"GroqRenderer initialized: model={self.model}")

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        deadline_s: Optional[float] = None,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream tokens from Groq API."""
        if max_tokens is None:
            max_tokens = self.tokens_default

        import requests

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "max_tokens": min(max_tokens, 2048),
                    "temperature": max(0.0, min(2.0, temperature)),
                },
                timeout=30,
            )
            if not response.ok:
                logger.error(
                    "Groq API error response: status=%s body=%s",
                    response.status_code,
                    response.text,
                )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    if data:
                        try:
                            import json
                            obj = json.loads(data)
                            if obj.get("choices"):
                                delta = obj["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    yield delta["content"]
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate non-streaming response."""
        chunks = list(self.stream(prompt, max_tokens=max_tokens, temperature=temperature))
        return "".join(chunks)
