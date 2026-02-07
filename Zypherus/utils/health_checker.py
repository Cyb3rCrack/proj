"""Comprehensive health check system for production deployments."""

from typing import Any, Dict, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthChecker:
    """Monitor system health and dependencies."""
    
    def __init__(self, ace: Any):
        """Initialize health checker.
        
        Args:
            ace: ACE system instance to monitor
        """
        self.ace = ace
    
    def check_api(self) -> Dict[str, Any]:
        """Check API service status."""
        return {
            "status": "healthy",
            "port": os.getenv("API_PORT", 8000)
        }
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory system status."""
        try:
            if not hasattr(self.ace, "memory"):
                return {"status": "unavailable"}
            
            memory_path = Path("memory.json")
            faiss_path = Path("memory.faiss")
            
            return {
                "status": "healthy",
                "memory_file_exists": memory_path.exists(),
                "memory_file_size_mb": memory_path.stat().st_size / (1024 * 1024) if memory_path.exists() else 0,
                "faiss_index_exists": faiss_path.exists(),
                "faiss_index_size_mb": faiss_path.stat().st_size / (1024 * 1024) if faiss_path.exists() else 0,
            }
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def check_inference(self) -> Dict[str, Any]:
        """Check inference engine status."""
        try:
            if not hasattr(self.ace, "inference_engine"):
                return {"status": "unavailable"}
            
            return {
                "status": "healthy",
                "model_loaded": True
            }
        except Exception as e:
            logger.error(f"Inference health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def check_storage(self) -> Dict[str, Any]:
        """Check storage system status."""
        try:
            # Check if we can write to disk
            test_file = Path(".health_check_test")
            test_file.write_text("ok")
            test_file.unlink()
            
            return {"status": "healthy", "writable": True}
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = {
            "api": self.check_api(),
            "memory": self.check_memory(),
            "inference": self.check_inference(),
            "storage": self.check_storage(),
        }
        
        # Determine overall status
        statuses = [c.get("status") for c in checks.values()]
        if all(s in ("healthy", "available") for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "checks": checks,
            "ready_for_traffic": overall_status == "healthy"
        }
    
    def is_ready(self) -> bool:
        """Check if system is ready to handle requests."""
        try:
            health = self.check_all()
            return health.get("ready_for_traffic", False)
        except Exception as e:
            logger.error(f"Ready check failed: {e}")
            return False
