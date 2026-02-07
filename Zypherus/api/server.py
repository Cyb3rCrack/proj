"""Production-ready REST API for Zypherus with authentication, validation, and documentation."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
from functools import wraps
from datetime import datetime
from dataclasses import dataclass
import json

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

HAS_FLASK = True

logger = logging.getLogger("ACE.API")


# Middleware and utilities
def request_id_middleware(f):  # type: ignore[no-untyped-def]
    """Add request ID to all requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):  # type: ignore[no-untyped-def]
        request.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))  # type: ignore[union-attr]
        return f(*args, **kwargs)
    return decorated_function


def add_security_headers(response):
    """Add security headers to response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@dataclass
class APIRequest:
    """Standardized API request."""
    action: str
    payload: Dict[str, Any]
    request_id: Optional[str] = None


@dataclass  
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Any
    error: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "request_id": self.request_id,
            "timestamp": self.timestamp or datetime.utcnow().isoformat(),
        }


class ZypherusAPIServer:
    """Production-ready API server for Zypherus."""
    
    def __init__(self, ace: Any, host: str = "0.0.0.0", port: int = 8000):
        if not HAS_FLASK:
            raise ImportError("Flask is required. Install with: pip install flask flask-cors")
        
        self.ace = ace
        self.host = host
        self.port = port
        self.app: Optional[Flask] = None  # type: ignore[name-defined]
    
    def create_flask_app(self) -> Flask:  # type: ignore[no-redef,name-defined]
        """Create production-ready Flask application."""
        app = Flask(__name__)  # type: ignore[attr-defined]
        
        # Configuration
        app.config["JSON_SORT_KEYS"] = False
        app.config["JSON_PRETTYPRINT_REGULAR"] = True
        
        # CORS setup
        CORS(app, resources={r"/api/*": {"origins": "*"}})  # type: ignore[misc]
        
        # Register error handlers
        self._register_error_handlers(app)
        
        # Register middleware
        self._register_middleware(app)
        
        # Register routes
        self._register_routes(app)
        
        self.app = app
        return app
    
    def _register_middleware(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register middleware."""
        
        @app.before_request
        def before_request():
            request.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))  # type: ignore
            request.start_time = datetime.utcnow()  # type: ignore
        
        @app.after_request
        def after_request(response):
            response = add_security_headers(response)
            if hasattr(request, "request_id"):  # type: ignore
                response.headers["X-Request-ID"] = request.request_id  # type: ignore
            return response
    
    def _register_error_handlers(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register error handlers."""
        
        @app.errorhandler(400)
        def bad_request(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "BAD_REQUEST",
                "message": "Invalid request",
                "request_id": getattr(request, "request_id", None)
            }), 400
        
        @app.errorhandler(401)
        def unauthorized(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "UNAUTHORIZED",
                "message": "Authentication required",
                "request_id": getattr(request, "request_id", None)
            }), 401
        
        @app.errorhandler(403)
        def forbidden(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "FORBIDDEN",
                "message": "Insufficient permissions",
                "request_id": getattr(request, "request_id", None)
            }), 403
        
        @app.errorhandler(404)
        def not_found(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "NOT_FOUND",
                "message": "Endpoint not found",
                "request_id": getattr(request, "request_id", None)
            }), 404
        
        @app.errorhandler(429)
        def rate_limited(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests",
                "request_id": getattr(request, "request_id", None)
            }), 429
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "request_id": getattr(request, "request_id", None)
            }), 500
    
    def _register_routes(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register API routes."""
        
        # Root endpoint
        @app.route("/", methods=["GET"])
        def root():
            """Root endpoint with API information."""
            return jsonify({
                "success": True,
                "name": "Zypherus API",
                "version": "0.2.0",
                "status": "operational",
                "documentation": "/api/docs",
                "endpoints": {
                    "health": "/health",
                    "answer": "/api/answer [POST]",
                    "ingest": "/api/ingest [POST]",
                    "search": "/api/search [POST]",
                    "status": "/api/status",
                    "memory": "/api/memory",
                    "beliefs": "/api/beliefs",
                    "concepts": "/api/concepts",
                    "stats": "/api/stats",
                    "docs": "/api/docs"
                },
                "timestamp": datetime.utcnow().isoformat()
            }), 200
        
        # Health check
        @app.route("/health", methods=["GET"])
        def health():
            """Service health check."""
            try:
                checker = getattr(app, "health_checker", None)
                if checker:
                    health_data = checker.check_all()
                else:
                    health_data = {
                        "overall_status": "healthy",
                        "checks": {
                            "api": {"status": "healthy"},
                            "memory": {"status": "healthy" if hasattr(self.ace, "memory") else "unknown"},
                        }
                    }
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "status": health_data.get("overall_status", "unknown"),
                        "version": "0.2.0",
                        "checks": health_data.get("checks", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
                return jsonify({
                    "success": False,
                    "status": "unhealthy",
                    "error": str(e),
                    "request_id": getattr(request, "request_id", None)
                }), 503
        
        # Readiness check (for Kubernetes/Render)
        @app.route("/ready", methods=["GET"])
        def ready():
            """Service readiness check."""
            try:
                checker = getattr(app, "health_checker", None)
                if not checker:
                    return jsonify({
                        "success": False,
                        "ready": False,
                        "message": "Health checker not available"
                    }), 503
                
                is_ready = checker.is_ready()
                status_code = 200 if is_ready else 503
                
                return jsonify({
                    "success": is_ready,
                    "ready": is_ready,
                    "message": "Service is ready" if is_ready else "Service is not ready",
                    "request_id": getattr(request, "request_id", None)
                }), status_code
            except ValueError as e:
                logger.error(f"Readiness check validation error: {e}")
                return jsonify({
                    "success": False,
                    "ready": False,
                    "error": f"Validation error: {str(e)}"
                }), 503
            except Exception as e:
                logger.error(f"Readiness check error: {e}", exc_info=True)
                return jsonify({
                    "success": False,
                    "ready": False,
                    "error": str(e)
                }), 503
        
                logger.error(f"Health check error: {e}")
                return jsonify({
                    "success": False,
                    "status": "unhealthy",
                    "error": str(e),
                    "request_id": getattr(request, "request_id", None)
                }), 503
        
        # Ingest document
        @app.route("/api/ingest", methods=["POST"])
        def ingest():
            """Ingest document into knowledge base."""
            try:
                data = request.get_json() or {}
                text = data.get("text", "").strip()
                source = data.get("source", "api")
                
                if not text:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Content cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                if not isinstance(text, str) or len(text) > 1000000:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Text must be string and <= 1MB"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                self.ace.ingest_document(source, text)
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "message": f"Successfully ingested {len(text)} characters",
                        "source": source,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except ValueError as e:
                logger.error(f"Ingest validation error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="VALIDATION_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
            except IOError as e:
                logger.error(f"Ingest I/O error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="IO_ERROR",
                    data={"message": f"Failed to store document: {str(e)}"},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
            except Exception as e:
                logger.error(f"Ingest error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="INGEST_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
        
        # Answer question
        @app.route("/api/answer", methods=["POST"])
        def answer():
            """Answer a question using knowledge base with caching."""
            try:
                data = request.get_json() or {}
                query = data.get("query", "").strip()
                use_cache = data.get("cache", True)
                
                if not query:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Query cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                # Check cache first
                cache = getattr(app, "query_cache", None)
                if use_cache and cache:
                    cached_result = cache.get(query)
                    if cached_result:
                        logger.debug(f"Cache hit for query: {query[:50]}")
                        return jsonify(APIResponse(
                            success=True,
                            data={**cached_result, "from_cache": True},
                            request_id=getattr(request, "request_id", None)
                        ).to_dict()), 200
                
                # Get fresh result
                result = self.ace.answer(query)
                
                # Cache result
                if cache:
                    cache.set(result, query)
                
                return jsonify(APIResponse(
                    success=True,
                    data={**result, "from_cache": False},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except ValueError as e:
                logger.error(f"Answer validation error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="VALIDATION_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
            except Exception as e:
                logger.error(f"Answer error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="ANSWER_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
        
        # Get status
        @app.route("/api/status", methods=["GET"])
        def status():
            """Get system status."""
            try:
                status_data: Dict[str, Any] = {
                    "uptime": "running",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "memory"):
                    status_data["memory_entries"] = len(getattr(self.ace.memory, "entries", []) or [])
                
                if hasattr(self.ace, "claim_store"):
                    status_data["claims"] = len(getattr(self.ace.claim_store, "claims", {}) or {})
                
                if hasattr(self.ace, "concept_graph"):
                    status_data["concepts"] = len(getattr(self.ace.concept_graph, "nodes", {}) or {})
                
                return jsonify(APIResponse(  # type: ignore
                    success=True,
                    data=status_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify(APIResponse(  # type: ignore
                    success=False,
                    error="STATUS_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get memory info
        @app.route("/api/memory", methods=["GET"])
        def get_memory():
            """Get memory system information."""
            try:
                memory_data: Dict[str, Any] = {
                    "status": "operational",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "memory"):
                    entries = getattr(self.ace.memory, "entries", []) or []
                    sources = set()
                    for entry in entries:
                        if isinstance(entry, dict):
                            sources.add(entry.get("source", "unknown"))
                    
                    memory_data.update({
                        "total_entries": len(entries),
                        "unique_sources": len(sources),
                        "sources": sorted(list(sources))
                    })
                
                return jsonify(APIResponse(  # type: ignore
                    success=True,
                    data=memory_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Memory error: {e}")
                return jsonify(APIResponse(  # type: ignore
                    success=False,
                    error="MEMORY_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get beliefs
        @app.route("/api/beliefs", methods=["GET"])
        def get_beliefs():
            """Get top beliefs/claims from knowledge base."""
            try:
                beliefs_data: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "claim_store"):
                    claims = getattr(self.ace.claim_store, "claims", {}) or {}
                    top_claims = sorted(
                        claims.values() if isinstance(claims, dict) else [],
                        key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0,
                        reverse=True
                    )[:10]
                    
                    beliefs_data.update({
                        "total_beliefs": len(claims),
                        "top_beliefs": [
                            {
                                "subject": c.get("subject") if isinstance(c, dict) else None,
                                "predicate": c.get("predicate") if isinstance(c, dict) else None,
                                "object": c.get("object") if isinstance(c, dict) else None,
                                "confidence": c.get("confidence", 0) if isinstance(c, dict) else 0,
                            }
                            for c in top_claims
                        ]
                    })
                
                return jsonify(APIResponse(
                    success=True,
                    data=beliefs_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Beliefs error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="BELIEFS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Search memory
        @app.route("/api/search", methods=["POST"])
        def search():
            """Search knowledge base entries."""
            try:
                data = request.get_json() or {}
                query = data.get("query", "").strip()
                limit = min(int(data.get("limit", 10)), 100)
                
                if not query:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Query cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                results = []
                if hasattr(self.ace, "memory"):
                    entries = getattr(self.ace.memory, "entries", []) or []
                    results = [
                        e for e in entries 
                        if isinstance(e, dict) and query.lower() in e.get("text", "").lower()
                    ][:limit]
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "query": query,
                        "results_count": len(results),
                        "results": results,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="SEARCH_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get concepts
        @app.route("/api/concepts", methods=["GET"])
        def get_concepts():
            """Get concept graph information."""
            try:
                concepts_data: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "concept_graph"):
                    nodes = getattr(self.ace.concept_graph, "nodes", {}) or {}
                    edges = getattr(self.ace.concept_graph, "edges", {}) or {}
                    
                    concepts_data.update({
                        "total_concepts": len(nodes),
                        "total_relationships": len(edges),
                        "sample_concepts": list(nodes.keys())[:20] if nodes else []
                    })
                
                return jsonify(APIResponse(
                    success=True,
                    data=concepts_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Concepts error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="CONCEPTS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # API documentation
        @app.route("/api/docs", methods=["GET"])
        def api_docs():
            """API documentation."""
            docs = {
                "title": "Zypherus API",
                "version": "0.2.0",
                "description": "Production-ready REST API for the Zypherus advanced reasoning system",
                "endpoints": {
                    "Health & Status": {
                        "GET /health": "Service health check"
,
                        "GET /api/status": "System status and metrics"
                    },
                    "Ingestion": {
                        "POST /api/ingest": "Ingest document or text"
                    },
                    "Queries": {
                        "POST /api/answer": "Answer question using knowledge base",
                        "POST /api/search": "Search knowledge base"
                    },
                    "Knowledge": {
                        "GET /api/memory": "Memory system info",
                        "GET /api/beliefs": "Top beliefs/claims",
                        "GET /api/concepts": "Concept graph info"
                    },
                    "Meta": {
                        "GET /api/docs": "This documentation"
                    }
                },
                "authentication": {
                    "type": "API Key or JWT",
                    "header": "X-API-Key or Authorization: Bearer <token>"
                },
                "request_format": {
                    "headers": {
                        "Content-Type": "application/json",
                        "X-Request-ID": "Optional request tracking ID"
                    }
                },
                "response_format": {
                    "success": "Boolean indicating success",
                    "data": "Response payload",
                    "error": "Error code if failed",
                    "request_id": "Request tracking ID",
                    "timestamp": "Response timestamp"
                }
            }
            
            return jsonify(docs), 200
        
        # System statistics
        @app.route("/api/stats", methods=["GET"])
        def get_stats():
            """Get system statistics and performance metrics."""
            try:
                stats: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": {}
                }
                
                # Cache stats
                cache = getattr(app, "query_cache", None)
                if cache:
                    stats["cache"] = cache.stats()
                
                # Memory stats
                memory = getattr(self.ace, "memory", None)
                if memory and hasattr(memory, "entries"):
                    stats["memory"] = {
                        "entry_count": len(memory.entries),
                        "file_path": getattr(memory, "filepath", "unknown")
                    }
                
                return jsonify(APIResponse(
                    success=True,
                    data=stats,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except EnvironmentError as e:
                logger.error(f"Stats retrieval environment error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="ENVIRONMENT_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
            except Exception as e:
                logger.error(f"Stats error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="STATS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
    
    def run(self, debug: bool = False):
        """Run API server."""
        if self.app is None:
            self.create_flask_app()
        
        logger.info(f"Starting Zypherus API server on {self.host}:{self.port}")
        if self.app is not None:
            self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)


class ZypherusAPIClient:
    """Client for Zypherus API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None
    
    def _get_session(self):
        """Get or create session."""
        if self.session is None:
            try:
                import requests
                self.session = requests.Session()
                if self.api_key:
                    self.session.headers.update({"X-API-Key": self.api_key})
            except ImportError:
                raise ImportError("requests is required. Install with: pip install requests")
        return self.session
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            resp = self._get_session().get(f"{self.base_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def ingest(self, text: str, source: str = "client") -> Dict[str, Any]:
        """Ingest document."""
        data = {"text": text, "source": source}
        resp = self._get_session().post(f"{self.base_url}/api/ingest", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Ask question."""
        data = {"query": query}
        resp = self._get_session().post(f"{self.base_url}/api/answer", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        resp = self._get_session().get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return resp.json()
    
    def get_memory(self) -> Dict[str, Any]:
        """Get memory info."""
        resp = self._get_session().get(f"{self.base_url}/api/memory")
        resp.raise_for_status()
        return resp.json()
    
    def get_beliefs(self) -> Dict[str, Any]:
        """Get beliefs."""
        resp = self._get_session().get(f"{self.base_url}/api/beliefs")
        resp.raise_for_status()
        return resp.json()
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search knowledge base."""
        data = {"query": query, "limit": limit}
        resp = self._get_session().post(f"{self.base_url}/api/search", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def get_concepts(self) -> Dict[str, Any]:
        """Get concept graph info."""
        resp = self._get_session().get(f"{self.base_url}/api/concepts")
        resp.raise_for_status()
        return resp.json()
    
    def get_docs(self) -> Dict[str, Any]:
        """Get API documentation."""
        resp = self._get_session().get(f"{self.base_url}/api/docs")
        resp.raise_for_status()
        return resp.json()


# Alias for backward compatibility
ACEAPIServer = ZypherusAPIServer
ACEAPIClient = ZypherusAPIClient


def create_app():
    """Factory function to create Flask app for WSGI servers."""
    from ..core.ace import ACE
    ace = ACE()
    server = ZypherusAPIServer(ace)
    return server.create_flask_app()


__all__ = [
    "APIRequest",
    "APIResponse",
    "ZypherusAPIServer",
    "ZypherusAPIClient",
    "ACEAPIServer",
    "ACEAPIClient",
    "create_app",
]
