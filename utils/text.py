"""Text/NLP utilities shared across ACE.

These are dependency-optional. If spaCy is unavailable or no model is installed,
helpers fall back to simple regex heuristics.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_NLP = None


def get_nlp():
    """Return a cached spaCy nlp pipeline if available, else None."""
    global _NLP
    if _NLP is not None:
        return _NLP

    if os.getenv("ACE_DISABLE_SPACY", "").strip().lower() in {"1", "true", "yes"}:
        _NLP = None
        return None

    try:
        import spacy  # type: ignore
    except Exception:
        _NLP = None
        return None

    for model in ("en_core_web_sm", "en_core_web_trf"):
        try:
            _NLP = spacy.load(model)
            return _NLP
        except Exception:
            continue

    _NLP = None
    return None


def answer_shape_issue(answer: str) -> Optional[str]:
    """Return a short reason string if the answer looks like a fragment."""
    if not isinstance(answer, str):
        return "non-text answer"
    s = answer.strip()
    if not s:
        return "empty answer"

    words = [w for w in re.findall(r"[A-Za-z0-9']+", s) if w]
    if len(words) < 3:
        return "too short"

    nlp = get_nlp()
    if nlp is not None:
        try:
            doc = nlp(s)
            has_verb = any(getattr(t, "pos_", "") in {"VERB", "AUX"} for t in doc)
            if not has_verb:
                return "no verb"
            return None
        except Exception:
            pass

    verb_like = re.search(
        r"\b(is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|will|would|should|may|might|must|means|mean|refers|refer|defined|define|consists|consist|includes|include|contains|contain|causes|cause|affects|affect|orbits|orbit)\b",
        s,
        flags=re.IGNORECASE,
    )
    if not verb_like:
        return "no verb"
    return None


__all__ = ["get_nlp", "answer_shape_issue"]
