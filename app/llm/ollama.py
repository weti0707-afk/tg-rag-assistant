from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 120.0

    async def chat(self, system: str, user: str) -> str:
        """
        Ollama /api/chat (stream=false)
        """
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()
