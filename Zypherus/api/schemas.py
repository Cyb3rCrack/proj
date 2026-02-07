"""Data validation schemas for Zypherus API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from flask import request


class SourceEnum(str, Enum):
    """Valid source types."""
    TEXT = "text"
    URL = "url"
    FILE = "file"
    YOUTUBE = "youtube"


class IngestRequest(BaseModel):
    """Ingest document request."""
    source_type: SourceEnum = Field(..., description="Type of source: text, url, file, or youtube")
    content: str = Field(..., min_length=1, description="Content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    
    @validator("content")
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "source_type": "text",
                "content": "Python is a programming language known for readability.",
                "metadata": {"author": "user123", "source": "tutorial"}
            }
        }


class QuestionRequest(BaseModel):
    """Question answering request."""
    query: str = Field(..., min_length=1, max_length=5000, description="Question to answer")
    include_sources: Optional[bool] = Field(default=True, description="Include source documents")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum results to return")
    confidence_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")
    
    @validator("query")
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How does Python handle memory management?",
                "include_sources": True,
                "max_results": 5,
                "confidence_threshold": 0.5
            }
        }


class SearchRequest(BaseModel):
    """Knowledge base search request."""
    query: str = Field(..., min_length=1, description="Search query")
    search_type: Optional[str] = Field(default="semantic", description="Search type: semantic, keyword, or hybrid")
    limit: Optional[int] = Field(default=10, ge=1, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "search_type": "hybrid",
                "limit": 10
            }
        }


class BeliefQuery(BaseModel):
    """Query beliefs/concepts in knowledge base."""
    entity: str = Field(..., min_length=1, description="Entity name to query")
    include_relationships: Optional[bool] = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "entity": "Python",
                "include_relationships": True
            }
        }


class MemoryAction(str, Enum):
    """Valid memory actions."""
    GET_STATS = "get_stats"
    CLEAR_CACHE = "clear_cache"
    REBUILD_INDEX = "rebuild_index"
    VALIDATE = "validate"


class MemoryRequest(BaseModel):
    """Memory management request."""
    action: MemoryAction = Field(..., description="Action to perform")
    layer: Optional[str] = Field(default=None, description="Specific layer to target")
    
    class Config:
        schema_extra = {
            "example": {
                "action": "get_stats",
                "layer": "semantic"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Check timestamp")
    components: Dict[str, str] = Field(..., description="Status of each component")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.2.0",
                "timestamp": "2026-02-06T10:30:00Z",
                "components": {
                    "memory": "healthy",
                    "embeddings": "healthy",
                    "reasoning": "healthy"
                }
            }
        }


class APIResponse(BaseModel):
    """Standard API response."""
    success: bool = Field(..., description="Whether request succeeded")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {"answer": "Python uses reference counting for memory management."},
                "error": None,
                "request_id": "req_12345"
            }
        }


class TokenRequest(BaseModel):
    """Token generation request."""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=8, description="Password")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "user@example.com",
                "password": "secure-password-123"
            }
        }


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "Bearer",
                "expires_in": 86400
            }
        }


def validate_request_schema(schema_class):
    """Decorator to validate request against schema."""
    def decorator(fn):
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                data = schema_class(**request.get_json() or {})
                request.validated_data = data  # type: ignore[attr-defined]
                return fn(*args, **kwargs)
            except ValueError as e:
                return {
                    "success": False,
                    "error": "Validation error",
                    "details": str(e)
                }, 400
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator
