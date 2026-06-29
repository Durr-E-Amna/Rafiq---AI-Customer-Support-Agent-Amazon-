"""
LLM client - thin wrapper around Groq's API.

Every agent node goes through this one place rather than calling the Groq
SDK directly, so swapping models, adding retry logic, or logging token
usage only has to happen here.
"""

import json
import os

from groq import Groq

MODEL = "llama-3.3-70b-versatile"


class GroqClient:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """For structured outputs (intent classification). Asks the model
        to return JSON only, then parses it defensively."""
        response = self.client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw}
