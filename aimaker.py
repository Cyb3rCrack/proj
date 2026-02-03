"""Backward-compatibility façade for the old monolithic `aimaker.py`.

All real implementations live under the `ace/` package. This module re-exports
the historical names so existing scripts keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("ACE")

# Extraction helpers
from ace.extraction.chunking import chunk_text
from ace.extraction.claims import extract_atomic_claims
from ace.extraction.parsing import parse_claim
from ace.extraction.concepts import extract_concepts

# LLM modules
from ace.llm.nli import NLIContradictionChecker
from ace.llm.renderer import LLMRenderer
from ace.llm.distiller import Distiller

# Beliefs + memory
from ace.beliefs.graph import BeliefGraph
from ace.beliefs.claims import ClaimStore, infer_domain
from ace.memory.index import KnowledgeIndex

# Concepts + reasoning
from ace.concepts.graph import ConceptGraph
from ace.inference.reasoning import ReasoningEngine

# Core orchestration/components
from ace.core.embedding import EmbeddingModule
from ace.core.question import QuestionInterpreter
from ace.core.dialogue import DialogueManager
from ace.core.decision import DecisionPolicy
from ace.core.ace import ACE


def extract_relations(text: str) -> List[Dict[str, str]]:
    # Relation extraction removed until implemented.
    return []


__all__ = [
    # Entry / orchestrator
    "ACE",
    # Core components
    "EmbeddingModule",
    "QuestionInterpreter",
    "DialogueManager",
    "DecisionPolicy",
    # LLM utilities
    "LLMRenderer",
    "NLIContradictionChecker",
    "Distiller",
    # Memory / beliefs
    "KnowledgeIndex",
    "ClaimStore",
    "BeliefGraph",
    # Concepts / reasoning
    "ConceptGraph",
    "ReasoningEngine",
    # Extraction helpers
    "chunk_text",
    "extract_atomic_claims",
    "parse_claim",
    "extract_concepts",
    "infer_domain",
    "extract_relations",
    # REPL
    "run_repl",
    "main",
]

def run_repl():
    # Delegates to the new modular CLI while we migrate code out of this file.
    from ace.cli.repl import run_repl as _run

    _run(ACE())


def main():
    run_repl()

if __name__ == "__main__":
    main()