import os
import logging
import signal
import sys
import atexit
from logging.config import dictConfig
from functools import wraps

# Configure structured logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for graceful shutdown
_app_state = {"shutting_down": False}


def graceful_shutdown(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _app_state["shutting_down"] = True
    
    try:
        # Save state before exit
        from Zypherus.core.ace import ACE
        ace = ACE()
        if hasattr(ace, "memory") and hasattr(ace.memory, "save"):
            logger.info("Saving memory to disk...")
            ace.memory.save()
            logger.info("Memory saved successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Shutdown complete")
    sys.exit(0)


def create_app():
    """Create Flask app with production configuration."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from Zypherus.api.server import ZypherusAPIServer
        from Zypherus.core.ace import ACE
        from Zypherus.utils.health_checker import HealthChecker
        from Zypherus.utils.query_cache import QueryCache
        from Zypherus.utils.request_validation import setup_request_validation
    except ImportError as e:
        logger.error(f"Failed to import Zypherus modules: {e}")
        raise
    
    logger.info("Initializing Zypherus ACE system...")
    ace = ACE()
    
    logger.info("Creating API server...")
    api_server = ZypherusAPIServer(
        ace,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )
    
    app = api_server.create_flask_app()
    
    # Setup query caching
    cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    app.query_cache = QueryCache(ttl_seconds=cache_ttl, max_size=cache_size)  # type: ignore[attr-defined]
    logger.info(f"Query cache initialized (TTL: {cache_ttl}s, max: {cache_size} entries)")
    
    # Setup health checker
    app.health_checker = HealthChecker(ace)  # type: ignore[attr-defined]
    logger.info("Health checker initialized")
    
    # Setup request validation
    max_size = int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10MB
    setup_request_validation(app, max_content_length=max_size)
    logger.info(f"Request validation enabled (max size: {max_size} bytes)")
    
    # Add shutdown check middleware
    @app.before_request
    def check_shutdown():
        """Check if app is shutting down."""
        from flask import jsonify
        if _app_state.get("shutting_down"):
            return jsonify({
                "success": False,
                "error": "SERVICE_SHUTTING_DOWN",
                "message": "Server is shutting down"
            }), 503
    
    # Add Sentry integration if available
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.flask import FlaskIntegration
            
            sentry_sdk.init(
                sentry_dsn,
                integrations=[FlaskIntegration()],
                traces_sample_rate=float(os.getenv("SENTRY_TRACE_RATE", "0.1"))
            )
            logger.info("Sentry error tracking initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed, skipping Sentry setup")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    
    logger.info("Zypherus API server ready")
    return app


if __name__ == "__main__":
    app = create_app()
    debug = os.getenv("FLASK_ENV") == "development"
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("GUNICORN_WORKERS", "4"))
    
    if debug:
        logger.info(f"Starting development server on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    else:
        logger.info(f"Use gunicorn to start production server: gunicorn wsgi:app -w {workers}")


# Create app instance for WSGI servers
app = create_app()
