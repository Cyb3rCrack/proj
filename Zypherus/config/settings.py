"""Centralized configuration system for ACE.

Supports environment variables, config files, and programmatic overrides.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger("ZYPHERUS.Config")


@dataclass
class MemoryConfig:
    """Memory/storage configuration."""
    max_total_entries: int = 50000  # Increased from 5000 - store everything valuable
    max_entries_per_source: int = 5000  # Increased from 800 - no artificial limits
    use_faiss: bool = True
    faiss_metric: str = "inner_product"  # inner_product (cosine) or l2
    embedding_dim: int = 384  # MiniLM dimension
    compression_enabled: bool = True
    compression_quality: int = 80  # 1-100, higher = more compression
    persistence_path: str = "memory.json"
    auto_save_interval_s: float = 300.0  # Auto-save every 5 min


@dataclass
class ClaimsConfig:
    """Belief/claims configuration."""
    max_claims: int = 10000
    decay_half_life_days: float = 45.0
    inference_weight: float = 0.60
    max_inferred_per_cycle: int = 25
    confidence_threshold_high: float = 0.70
    confidence_threshold_medium: float = 0.50
    confidence_threshold_low: float = 0.30
    nli_contradiction_threshold: float = 0.80
    persistence_path: str = "data/knowledge/claims.json"
    auto_save_interval_s: float = 300.0


@dataclass
class DistillerConfig:
    """Knowledge distillation configuration."""
    max_facts: int = 10
    fact_char_limit: int = 500
    summary_char_limit: int = 300
    max_entities: int = 20
    workers: int = 3
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 100
    max_tokens_extraction: int = 500
    timeout_s: Optional[float] = None


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True
    cache_embeddings: bool = True
    cache_max_size: int = 1000
    batch_size: int = 32
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class FastIngestConfig:
    """Fast ingestion configuration for high-throughput ingestion."""
    enabled: bool = True
    profile: str = "safe"  # "safe" or "max"
    chunk_size_default: int = 600
    chunk_size_web: int = 900
    skip_distiller: bool = True
    skip_relationships: bool = True
    skip_entities: bool = True
    skip_fact_extraction: bool = True


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    provider: str = "ollama"  # "ollama", "openai", "local", "offline"
    model_name: str = "neural-chat"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_delay_s: float = 2.0
    fallback_mode: str = "heuristic"  # "heuristic", "passthrough", "error"
    cache_enabled: bool = True
    cache_ttl_s: float = 3600.0
    memory_only: bool = True  # Refuse to answer without memory
    require_citations: bool = True  # Validate answers ground in sources
    allow_clarification_questions: bool = False  # Strict refusal mode


@dataclass
class DialogueConfig:
    """Conversation management configuration."""
    max_turns: int = 6
    max_history_chars: int = 5000
    coherence_check_enabled: bool = True
    repair_strategy: str = "clarify"  # "clarify", "ignore", "retry"
    context_window_size: int = 3


@dataclass
class ConceptGraphConfig:
    """Concept graph configuration."""
    max_nodes: int = 50000
    max_edges_per_node: int = 100
    relationship_types: list = field(default_factory=lambda: [
        "is-a", "part-of", "causes", "similar-to", "opposite-of", "related-to"
    ])
    cooccurrence_threshold: float = 0.3
    semantic_similarity_threshold: float = 0.6


@dataclass
class ExtractionConfig:
    """Text extraction configuration."""
    chunk_size: int = 500
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    sentence_splitter: str = "spacy"  # "spacy" or "regex"
    use_nlp_for_concepts: bool = True
    nlp_model: str = "en_core_web_sm"


@dataclass
class ReasoningConfig:
    """Symbolic reasoning configuration."""
    enable_causal_reasoning: bool = True
    enable_contradiction_detection: bool = True
    enable_domain_reasoning: bool = True
    max_reasoning_depth: int = 5
    confidence_propagation_factor: float = 0.85
    rule_conflict_strategy: str = "weighted_majority"  # "weighted_majority", "highest_confidence", "abstain"


@dataclass
class AnsweringConfig:
    """Answering behavior defaults for web-assisted QA."""
    default_mode: str = "balanced"  # fast | balanced | deep
    default_style: Optional[str] = None  # authoritative | practical | None
    recent_only: bool = False
    cache_min_confidence: float = 0.70
    cache_ttl_days: Dict[str, int] = field(default_factory=lambda: {
        "fast": 7,
        "balanced": 30,
        "deep": 90,
    })


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    file_size_mb: int = 10
    backup_count: int = 5
    telemetry_enabled: bool = True


@dataclass
class ACEConfig:
    """Master configuration for ACE."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    claims: ClaimsConfig = field(default_factory=ClaimsConfig)
    distiller: DistillerConfig = field(default_factory=DistillerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    fast_ingest: FastIngestConfig = field(default_factory=FastIngestConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    concept_graph: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    answering: AnsweringConfig = field(default_factory=AnsweringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Config saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    @classmethod
    def load(cls, path: str) -> ACEConfig:
        """Load config from JSON file."""
        try:
            if not os.path.exists(path):
                logger.warning(f"Config file not found: {path}, using defaults")
                return cls()
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            config = cls()
            for key, value in data.items():
                if hasattr(config, key) and isinstance(value, dict):
                    # Recursively update dataclass fields
                    setattr(config, key, cls._update_dataclass(getattr(config, key), value))
            
            logger.info(f"Config loaded from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()

    @staticmethod
    def _update_dataclass(obj: Any, data: Dict[str, Any]) -> Any:
        """Recursively update dataclass from dict."""
        if not isinstance(data, dict):
            return obj
        
        for key, value in data.items():
            if hasattr(obj, key):
                current = getattr(obj, key)
                if hasattr(current, '__dataclass_fields__'):
                    setattr(obj, key, ACEConfig._update_dataclass(current, value))
                else:
                    setattr(obj, key, value)
        return obj

    @classmethod
    def from_env(cls) -> ACEConfig:
        """Load config from environment variables with safe parsing."""
        config = cls()
        
        # Helper functions for safe parsing
        def safe_int(key: str, default: Optional[int] = None) -> Optional[int]:
            val = os.getenv(key)
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    logger.warning(f"Invalid int for {key}={val}, using default")
                    return default
            return default
        
        def safe_bool(key: str, default: bool = False) -> bool:
            val = os.getenv(key)
            if val is not None:
                return val.lower() in ("1", "true", "yes", "on")
            return default
        
        def safe_str(key: str, default: Optional[str] = None) -> Optional[str]:
            val = os.getenv(key)
            return val if val is not None else default
        
        # Memory config from env
        if (max_entries := safe_int("ZYPHERUS_MAX_ENTRIES")) is not None:
            config.memory.max_total_entries = max_entries
        if (max_per_source := safe_int("ZYPHERUS_MAX_ENTRIES_PER_SOURCE")) is not None:
            config.memory.max_entries_per_source = max_per_source
        config.memory.use_faiss = safe_bool("ZYPHERUS_USE_FAISS", config.memory.use_faiss)
        
        # LLM config from env
        if (provider := safe_str("ZYPHERUS_LLM_PROVIDER")) is not None:
            config.llm.provider = provider
        if (model := safe_str("ZYPHERUS_LLM_MODEL")) is not None:
            config.llm.model_name = model
        if (url := safe_str("ZYPHERUS_LLM_BASE_URL")) is not None:
            config.llm.base_url = url
        if (key := safe_str("ZYPHERUS_LLM_API_KEY")) is not None:
            config.llm.api_key = key

        # Embedding config from env
        if (model := safe_str("ZYPHERUS_EMBEDDING_MODEL")) is not None:
            config.embedding.model_name = model
        if (device := safe_str("ZYPHERUS_EMBEDDING_DEVICE")) is not None:
            config.embedding.device = device
        if (batch_size := safe_int("ZYPHERUS_EMBEDDING_BATCH_SIZE")) is not None:
            config.embedding.batch_size = batch_size
        config.embedding.normalize = safe_bool("ZYPHERUS_EMBEDDING_NORMALIZE", config.embedding.normalize)

        # Fast ingest config from env
        config.fast_ingest.enabled = safe_bool("ZYPHERUS_INGEST_FAST", config.fast_ingest.enabled)
        if (profile := safe_str("ZYPHERUS_INGEST_PROFILE")) is not None:
            config.fast_ingest.profile = profile
        if (chunk_def := safe_int("ZYPHERUS_INGEST_CHUNK_DEFAULT")) is not None:
            config.fast_ingest.chunk_size_default = chunk_def
        if (chunk_web := safe_int("ZYPHERUS_INGEST_CHUNK_WEB")) is not None:
            config.fast_ingest.chunk_size_web = chunk_web
        config.fast_ingest.skip_distiller = safe_bool("ZYPHERUS_INGEST_SKIP_DISTILLER", config.fast_ingest.skip_distiller)
        config.fast_ingest.skip_relationships = safe_bool("ZYPHERUS_INGEST_SKIP_RELATIONSHIPS", config.fast_ingest.skip_relationships)
        config.fast_ingest.skip_entities = safe_bool("ZYPHERUS_INGEST_SKIP_ENTITIES", config.fast_ingest.skip_entities)
        config.fast_ingest.skip_fact_extraction = safe_bool("ZYPHERUS_INGEST_SKIP_FACTS", config.fast_ingest.skip_fact_extraction)
        
        # Logging from env
        if (level := safe_str("ZYPHERUS_LOG_LEVEL")) is not None:
            config.logging.level = level
        if (filepath := safe_str("ZYPHERUS_LOG_FILE")) is not None:
            config.logging.file_path = filepath
        
        logger.info("Config loaded from environment variables")
        return config


def get_config(config_path: Optional[str] = None, use_env: bool = True) -> ACEConfig:
    """Get ACE configuration.
    
    Priority: env vars > config file > defaults
    """
    config = ACEConfig()
    
    # Load from config file if provided
    if config_path and os.path.exists(config_path):
        file_config = ACEConfig.load(config_path)
        config = file_config
    
    # Override with environment variables
    if use_env:
        env_config = ACEConfig.from_env()
        # Merge env config into loaded config
        for key in asdict(env_config):
            if getattr(env_config, key) != getattr(ACEConfig(), key):
                setattr(config, key, getattr(env_config, key))
    
    return config


__all__ = [
    "ACEConfig",
    "MemoryConfig",
    "ClaimsConfig",
    "DistillerConfig",
    "EmbeddingConfig",
    "FastIngestConfig",
    "LLMConfig",
    "DialogueConfig",
    "ConceptGraphConfig",
    "ExtractionConfig",
    "ReasoningConfig",
    "LoggingConfig",
    "get_config",
]
