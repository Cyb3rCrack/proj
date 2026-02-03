"""Distillation helpers.

Uses the LLM for summary/fact extraction when needed, and prefers spaCy NER
when available.
"""

from __future__ import annotations

from typing import Any, List

import requests

from ace.utils.text import get_nlp


class Distiller:
    def __init__(self, llm: Any):
        self.llm = llm

    def extract_facts(self, text: str, source: str) -> List[str]:
        prompt = f"""
Extract clear, factual bullet points from the text below.
Rules:
- Use only information in the text
- No speculation
- Short, concrete facts

Text:
{(text or '')[:4000]}

Facts (one per line):
"""
        try:
            r = requests.post(
                f"{self.llm.base_url}/api/generate",
                json={"model": self.llm.model, "prompt": prompt, "stream": False, "max_tokens": 400},
                timeout=60,
            )
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            lines = [l.strip().lstrip("-•*") for l in out.splitlines() if l.strip()]
            return lines
        except Exception:
            return []

    def extract_summary(self, text: str, max_len: int = 200) -> str:
        prompt = f"""
Write a concise summary (one paragraph) of the text below. Keep under {max_len} characters.

Text:
{(text or '')[:4000]}

Summary:
"""
        try:
            r = requests.post(
                f"{self.llm.base_url}/api/generate",
                json={"model": self.llm.model, "prompt": prompt, "stream": False, "max_tokens": 200},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            return ""

    def extract_entities(self, text: str) -> List[str]:
        if not text:
            return []
        nlp = get_nlp()
        if nlp is not None:
            try:
                doc = nlp(text)
                ents = [ent.text for ent in doc.ents]
                uniq = []
                for e in ents:
                    if e and e not in uniq:
                        uniq.append(e)
                return uniq
            except Exception:
                pass

        prompt = f"""
Extract named entities (people, places, organizations, topics) from the text below as a comma-separated list.

Text:
{(text or '')[:4000]}

Entities:
"""
        try:
            r = requests.post(
                f"{self.llm.base_url}/api/generate",
                json={"model": self.llm.model, "prompt": prompt, "stream": False, "max_tokens": 200},
                timeout=60,
            )
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            parts = [p.strip() for p in out.replace("\n", ",").split(",") if p.strip()]
            return parts
        except Exception:
            return []


__all__ = ["Distiller"]
