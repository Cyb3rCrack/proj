"""Type hints and validation utilities for ACE.

Provides:
- Common type aliases
- Type validation functions
- Runtime type checking decorators
"""

from __future__ import annotations

from typing import (
    Any, Dict, List, Set, Tuple, Optional, Union,
    Callable, Type, TypeVar, Generic, Protocol, Sequence
)
from functools import wraps
from logging import getLogger

logger = getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Common type aliases
ClaimType = Dict[str, Any]
ConceptType = str
EntityType = Dict[str, str]
EmbeddingType = List[float]
RelationType = Tuple[str, str, str]  # (subject, predicate, object)
ConfidenceScore = float  # 0.0 to 1.0
TimestampType = float  # Unix timestamp
TextType = str

# Complex types
QueryResult = Dict[str, Union[List[ClaimType], ConfidenceScore, int]]
DialogueTurnType = Dict[str, Union[str, float, Optional[str]]]
ReasoningResult = Dict[str, Any]  # Known, uncertain, contradictions, conclusion
MemorySnapshot = Dict[str, Union[List[Any], Dict[str, Any]]]

# Protocol types for structural subtyping
class EmbeddingProvider(Protocol):
    """Something that can provide embeddings."""
    def embed(self, text: str) -> EmbeddingType:
        """Generate embedding for text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingType]:
        """Generate embeddings for multiple texts."""
        ...


class Persister(Protocol):
    """Something that can persist and load data."""
    def save(self, data: Any, path: str) -> bool:
        """Save data to path."""
        ...
    
    def load(self, path: str) -> Optional[Any]:
        """Load data from path."""
        ...


def validate_type(expected_type: Type[T]) -> Callable:
    """Decorator to validate argument types at runtime."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for arg in args:
                if arg is not None and not isinstance(arg, expected_type):
                    logger.warning(
                        f"{func.__name__} received {type(arg)} but expected {expected_type}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensure_list(value: Union[T, List[T]]) -> List[T]:
    """Ensure value is a list."""
    return value if isinstance(value, list) else [value]


def ensure_dict(value: Union[Dict[K, V], Sequence[Tuple[K, V]]]) -> Dict[K, V]:
    """Ensure value is a dictionary."""
    if isinstance(value, dict):
        return value
    return dict(value)


def validate_claim(claim: ClaimType) -> bool:
    """Validate claim structure."""
    required_keys = {"subject", "predicate", "object"}
    return isinstance(claim, dict) and required_keys.issubset(claim.keys())


def validate_embedding(embedding: Any) -> bool:
    """Validate embedding structure."""
    return (
        isinstance(embedding, (list, tuple)) and
        len(embedding) > 0 and
        all(isinstance(x, (int, float)) for x in embedding)
    )


def validate_confidence(score: Any) -> bool:
    """Validate confidence score (0.0 to 1.0)."""
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0


def safe_type_cast(value: Any, target_type: Type[T], default: Optional[T] = None) -> Optional[T]:
    """Safely cast value to target type or return default."""
    try:
        if target_type == str:
            return str(value) if value is not None else default
        elif target_type == int:
            return int(value) if value is not None else default
        elif target_type == float:
            return float(value) if value is not None else default
        elif target_type == bool:
            return bool(value) if value is not None else default
        elif target_type == list:
            return list(value) if value is not None else default
        elif target_type == dict:
            return dict(value) if isinstance(value, dict) else default
        else:
            return value if isinstance(value, target_type) else default
    except (ValueError, TypeError):
        return default


class TypedDict(Generic[K, V]):
    """Simple typed dictionary wrapper."""
    
    def __init__(self, key_type: Type[K], value_type: Type[V]):
        self.key_type = key_type
        self.value_type = value_type
        self._data: Dict[K, V] = {}
    
    def __setitem__(self, key: K, value: V) -> None:
        if not isinstance(key, self.key_type):
            raise TypeError(f"Key must be {self.key_type}, got {type(key)}")
        if not isinstance(value, self.value_type):
            raise TypeError(f"Value must be {self.value_type}, got {type(value)}")
        self._data[key] = value
    
    def __getitem__(self, key: K) -> V:
        return self._data[key]
    
    def __contains__(self, key: K) -> bool:
        return key in self._data
    
    def items(self):
        return self._data.items()
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()


__all__ = [
    # Type aliases
    "ClaimType", "ConceptType", "EntityType", "EmbeddingType", "RelationType",
    "ConfidenceScore", "TimestampType", "TextType",
    "QueryResult", "DialogueTurnType", "ReasoningResult", "MemorySnapshot",
    
    # Protocols
    "EmbeddingProvider", "Persister",
    
    # Functions
    "validate_type", "ensure_list", "ensure_dict",
    "validate_claim", "validate_embedding", "validate_confidence",
    "safe_type_cast",
    
    # Classes
    "TypedDict",
]
