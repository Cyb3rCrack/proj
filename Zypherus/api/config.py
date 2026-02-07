"""Configuration management for Zypherus."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "json")
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    sentry_enabled: bool = os.getenv("SENTRY_ENABLED", "false").lower() == "true"
    sentry_environment: str = os.getenv("SENTRY_ENVIRONMENT", "production")


@dataclass
class APIConfig:
    """API configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    workers: int = int(os.getenv("WORKERS", "4"))
    
    # CORS
    cors_origins: list = [s.strip() for s in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]
    cors_allow_credentials: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds
    
    # Request timeout
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds


@dataclass
class SecurityConfig:
    """Security configuration."""
    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-prod")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiry_hours: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    
    # API Keys
    require_auth: bool = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
    
    # Headers
    security_headers_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year


@dataclass
class ModelConfig:
    """Model configuration."""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    device: str = os.getenv("DEVICE", "cpu")


@dataclass
class MemoryConfig:
    """Memory configuration."""
    max_memory_mb: int = int(os.getenv("MAX_MEMORY_MB", "2048"))
    enable_persistence: bool = os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true"
    persistence_path: str = os.getenv("PERSISTENCE_PATH", "./data/memory")
    
    # Layer sizes
    raw_capture_max_items: int = int(os.getenv("RAW_CAPTURE_MAX", "10000"))
    structural_layer_max_items: int = int(os.getenv("STRUCTURAL_LAYER_MAX", "5000"))
    semantic_layer_max_items: int = int(os.getenv("SEMANTIC_LAYER_MAX", "1000"))


@dataclass
class FeatureFlags:
    """Feature flags."""
    enable_web_ingestion: bool = os.getenv("FEATURE_WEB_INGESTION", "true").lower() == "true"
    enable_youtube_ingestion: bool = os.getenv("FEATURE_YOUTUBE_INGESTION", "true").lower() == "true"
    enable_reasoning: bool = os.getenv("FEATURE_REASONING", "true").lower() == "true"
    enable_belief_propagation: bool = os.getenv("FEATURE_BELIEF_PROP", "true").lower() == "true"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.logging = LoggingConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.model = ModelConfig()
        self.memory = MemoryConfig()
        self.features = FeatureFlags()
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.version = "0.2.0"
    
    def validate(self):
        """Validate configuration."""
        errors = []
        
        if self.security.require_auth and not self.security.jwt_secret:
            errors.append("JWT_SECRET_KEY required when REQUIRE_AUTH=true")
        
        if self.api.rate_limit_requests <= 0:
            errors.append("RATE_LIMIT_REQUESTS must be positive")
        
        if self.memory.max_memory_mb < 512:
            errors.append("MAX_MEMORY_MB should be at least 512")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    def to_dict(self):
        """Convert to dictionary (safe for API responses)."""
        return {
            "environment": self.environment,
            "version": self.version,
            "features": {
                "web_ingestion": self.features.enable_web_ingestion,
                "youtube_ingestion": self.features.enable_youtube_ingestion,
                "reasoning": self.features.enable_reasoning,
                "belief_propagation": self.features.enable_belief_propagation,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "rate_limiting": self.api.rate_limit_enabled,
            }
        }


def get_config() -> Config:
    """Get global configuration."""
    return _config


_config = Config()
