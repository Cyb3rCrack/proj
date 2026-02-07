"""Zypherus - Advanced AI Engine with Engineering Judgment"""

from __future__ import annotations

__version__ = "0.2.0"

# Core orchestration
from .core.ace import ACE
from .core.ace import ACE as ACEEnhanced

# Core components
from .core.embedding import EmbeddingModule
from .core.question import QuestionInterpreter
from .core.dialogue import DialogueManager, DialogueContext, DialogueState, IntentType
from .core.decision import DecisionPolicy

# LLM utilities
from .llm.renderer import LLMRenderer
from .llm.nli import NLIContradictionChecker
from .llm.distiller import Distiller
from .llm.fallback import LLMManager, create_llm_manager, LLMStrategy

# Memory / beliefs
from .memory.index import KnowledgeIndex
from .memory.optimization import EmbeddingIndex, MemoryOptimizer
from .memory.persistence import PersistenceManager, Checkpoint, CorruptionDetector, CompressionManager
from .beliefs.claims import ClaimStore, infer_domain
from .beliefs.graph import BeliefGraph

# Concepts / reasoning
from .concepts.graph import ConceptGraph, ConceptNode, ConceptEdge, RelationType
from .inference.reasoning import ReasoningEngine, SymbolicRule, ConfidenceLevel

# Extraction helpers
from .extraction.chunking import chunk_text
from .extraction.claims import extract_atomic_claims
from .extraction.parsing import parse_claim
from .extraction.concepts import extract_concepts

# Configuration
from .config.settings import get_config, ACEConfig

# Error handling & utilities
from .utils.errors_enhanced import setup_logging, safe_call, timed_operation, log_exception
from .utils.typing import (
    ClaimType, ConceptType, EntityType, EmbeddingType,
    ConfidenceScore, validate_claim, validate_embedding,
    ensure_list, ensure_dict, safe_type_cast
)
from .utils.ingestion_filter import IngestionFilter
from .utils.memory_cleaner import MemoryCleaner

# Async utilities
from .utils.async_utils import AsyncExecutor, AsyncBatcher, parallel_map, RateLimiter

# API
from .api import ACEAPIServer, ACEAPIClient

__all__ = [
    # Version
    "__version__",
    
    # Core orchestration
    "ACE",
    "ACEEnhanced",
    
    # Core components
    "EmbeddingModule",
    "QuestionInterpreter",
    "DialogueManager",
    "DialogueContext",
    "DialogueState",
    "IntentType",
    "DecisionPolicy",
    
    # LLM utilities
    "LLMRenderer",
    "NLIContradictionChecker",
    "Distiller",
    "LLMManager",
    "create_llm_manager",
    "LLMStrategy",
    
    # Memory / beliefs
    "KnowledgeIndex",
    "EmbeddingIndex",
    "MemoryOptimizer",
    "PersistenceManager",
    "Checkpoint",
    "CorruptionDetector",
    "CompressionManager",
    "ClaimStore",
    "infer_domain",
    "BeliefGraph",
    
    # Concepts / reasoning
    "ConceptGraph",
    "ConceptNode",
    "ConceptEdge",
    "RelationType",
    "ReasoningEngine",
    "SymbolicRule",
    "ConfidenceLevel",
    
    # Extraction helpers
    "chunk_text",
    "extract_atomic_claims",
    "parse_claim",
    "extract_concepts",
    
    # Configuration
    "get_config",
    "ACEConfig",
    
    # Error handling & utilities
    "setup_logging",
    "safe_call",
    "timed_operation",
    "log_exception",
    "IngestionFilter",
    "MemoryCleaner",
    
    # Type hints
    "ClaimType",
    "ConceptType",
    "EntityType",
    "EmbeddingType",
    "ConfidenceScore",
    "validate_claim",
    "validate_embedding",
    "ensure_list",
    "ensure_dict",
    "safe_type_cast",
    
    # Async utilities
    "AsyncExecutor",
    "AsyncBatcher",
    "parallel_map",
    "RateLimiter",
    
    # API
    "ACEAPIServer",
    "ACEAPIClient",
]
