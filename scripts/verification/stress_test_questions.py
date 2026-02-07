"""Adversarial question stress test harness.

Run manually to probe retrieval, disagreement handling, and self-critique.
"""

from __future__ import annotations

import json
import time
from typing import List, Dict

from Zypherus.core.ace import ACE


QUESTIONS: List[Dict[str, str]] = [
    {"q": "Tell me about the dark web", "mode": "balanced"},
    {"q": "What is zero trust security?", "mode": "fast"},
    {"q": "Compare TLS 1.2 vs TLS 1.3", "mode": "deep"},
    {"q": "Is SHA-1 still safe for signatures?", "mode": "balanced"},
    {"q": "How does Linux memory overcommit work?", "mode": "deep"},
    {"q": "Explain DNS over HTTPS risks", "mode": "balanced"},
    {"q": "What changed in HTTP/3?", "mode": "balanced"},
    {"q": "Is AES-256 broken?", "mode": "fast"},
    {"q": "What is the best password manager?", "mode": "balanced"},
    {"q": "What is the capital of Czechoslovakia?", "mode": "fast"},
]


def run():
    ace = ACE()
    results = []

    for item in QUESTIONS:
        q = item["q"]
        mode = item.get("mode", "balanced")
        print(f"\n[STRESS] Q: {q} (mode={mode})")
        start = time.time()
        res = ace.answer_with_web(q, mode=mode, recent_only=False, use_cache=False, store_pages=False)
        elapsed = time.time() - start

        results.append({
            "question": q,
            "mode": mode,
            "elapsed_s": round(elapsed, 2),
            "answer": res.get("answer"),
            "confidence": res.get("confidence"),
            "reason": res.get("reason"),
            "caveat": res.get("caveat"),
            "self_critique": res.get("self_critique"),
            "disagreements": res.get("disagreements"),
            "sources": res.get("sources"),
            "web_sources": res.get("web_sources"),
        })

    out_path = "data/verification/stress_test_results.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\n[STRESS] Results saved to {out_path}")
    except Exception as e:
        print(f"\n[STRESS] Failed to save results: {e}")


if __name__ == "__main__":
    run()
