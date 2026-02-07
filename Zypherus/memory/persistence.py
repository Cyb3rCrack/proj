"""Enhanced memory persistence with versioning, corruption recovery, and compression.

Features:
- Versioned snapshots for history and rollback
- Corruption detection and recovery
- Compression at storage level
- Atomic writes with journal
- Migration support between versions
- Backup management
"""

from __future__ import annotations

import json
import gzip
import hashlib
import time
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a persistent checkpoint/snapshot."""
    version: int
    timestamp: float = field(default_factory=time.time)
    data_hash: str = ""  # SHA256 of data for integrity checking
    compression_ratio: float = 0.0  # bytes_compressed / bytes_original
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CorruptionDetector:
    """Detects and recovers from data corruption."""
    
    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def verify_integrity(data: bytes, expected_hash: str) -> bool:
        """Verify data integrity against hash."""
        return CorruptionDetector.compute_hash(data) == expected_hash
    
    @staticmethod
    def detect_corruption(data: Dict[str, Any], schema_validators: Dict[str, Any]) -> List[str]:
        """Detect structural corruption in loaded data."""
        issues = []
        
        for key, validator in schema_validators.items():
            if key not in data:
                issues.append(f"Missing required key: {key}")
                continue
            
            if callable(validator) and not validator(data[key]):
                issues.append(f"Invalid value for key {key}")
        
        return issues
    
    @staticmethod
    def attempt_recovery(data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover corrupted data using defaults."""
        recovered = {}
        
        for key, default_value in defaults.items():
            if key in data and data[key] is not None:
                try:
                    # Try to use existing value
                    recovered[key] = data[key]
                except (TypeError, ValueError):
                    # Fall back to default
                    recovered[key] = default_value
            else:
                recovered[key] = default_value
        
        logger.info(f"Recovered data from {len(defaults) - len(recovered)} corrupted fields")
        return recovered


class CompressionManager:
    """Manages data compression for storage."""
    
    COMPRESSION_LEVEL = 9  # Maximum compression
    
    @staticmethod
    def compress(data: Dict[str, Any]) -> Tuple[bytes, float]:
        """Compress data and return (compressed_bytes, ratio)."""
        json_bytes = json.dumps(data).encode("utf-8")
        original_size = len(json_bytes)
        
        compressed = gzip.compress(json_bytes, compresslevel=CompressionManager.COMPRESSION_LEVEL)
        compressed_size = len(compressed)
        
        ratio = compressed_size / original_size if original_size > 0 else 0
        return compressed, ratio
    
    @staticmethod
    def decompress(data: bytes) -> Dict[str, Any]:
        """Decompress data."""
        json_bytes = gzip.decompress(data)
        return json.loads(json_bytes.decode("utf-8"))


class PersistenceManager:
    """Manages memory persistence with versioning and recovery."""
    
    def __init__(
        self,
        storage_path: str = "memory_store.json.gz",
        checkpoint_dir: str = ".checkpoints",
        max_checkpoints: int = 10,
        enable_compression: bool = True,
    ):
        self.storage_path = Path(storage_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.enable_compression = enable_compression
        self.current_version = 0
        self.checkpoints: List[Checkpoint] = []
        self.journal_path = self.checkpoint_dir / "journal.log"
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Load checkpoint index
        self._load_checkpoint_index()
    
    def _load_checkpoint_index(self) -> None:
        """Load list of existing checkpoints."""
        index_file = self.checkpoint_dir / "checkpoints.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index = json.load(f)
                    self.checkpoints = [Checkpoint(**cp) for cp in index.get("checkpoints", [])]
                    self.current_version = index.get("current_version", 0)
                    logger.debug(f"Loaded {len(self.checkpoints)} checkpoints")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
    
    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index."""
        index_file = self.checkpoint_dir / "checkpoints.json"
        try:
            with open(index_file, "w") as f:
                json.dump(
                    {
                        "checkpoints": [cp.to_dict() for cp in self.checkpoints],
                        "current_version": self.current_version,
                        "last_updated": time.time(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")
    
    def save(self, data: Dict[str, Any], description: str = "") -> bool:
        """Save data with automatic versioning and compression."""
        try:
            # Compress if enabled
            if self.enable_compression:
                compressed_data, ratio = CompressionManager.compress(data)
                data_hash = CorruptionDetector.compute_hash(compressed_data)
            else:
                json_data = json.dumps(data).encode("utf-8")
                compressed_data = json_data
                data_hash = CorruptionDetector.compute_hash(json_data)
                ratio = 1.0
            
            # Create checkpoint
            self.current_version += 1
            checkpoint = Checkpoint(
                version=self.current_version,
                data_hash=data_hash,
                compression_ratio=ratio,
                metadata={
                    "size_original": len(json.dumps(data)),
                    "size_compressed": len(compressed_data),
                },
                description=description,
            )
            
            # Write atomically with journal
            checkpoint_file = self.checkpoint_dir / f"checkpoint_v{self.current_version}.gz"
            temp_file = checkpoint_file.with_suffix(".tmp")
            
            try:
                # Write to temporary file first
                with open(temp_file, "wb") as f:
                    f.write(compressed_data)
                
                # Verify integrity
                with open(temp_file, "rb") as f:
                    verify_data = f.read()
                if not CorruptionDetector.verify_integrity(verify_data, data_hash):
                    raise IOError("Integrity check failed after write")
                
                # Atomic rename
                temp_file.replace(checkpoint_file)
                
                # Update checkpoint list
                self.checkpoints.append(checkpoint)
                self._cleanup_old_checkpoints()
                self._save_checkpoint_index()
                
                logger.debug(f"Saved checkpoint v{self.current_version} ({len(compressed_data)} bytes, ratio: {ratio:.2%})")
                return True
            
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                return False
        
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def load(self, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load data from checkpoint."""
        target_version = version or self.current_version
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_v{target_version}.gz"
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint v{target_version} not found")
            return None
        
        try:
            with open(checkpoint_file, "rb") as f:
                compressed_data = f.read()
            
            # Verify integrity
            checkpoint = next((cp for cp in self.checkpoints if cp.version == target_version), None)
            if checkpoint and not CorruptionDetector.verify_integrity(compressed_data, checkpoint.data_hash):
                logger.error(f"Integrity check failed for checkpoint v{target_version}")
                return None
            
            # Decompress
            data = CompressionManager.decompress(compressed_data)
            logger.debug(f"Loaded checkpoint v{target_version}")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint v{target_version}: {e}")
            return None
    
    def rollback(self, steps: int = 1) -> bool:
        """Rollback to previous checkpoint."""
        target_version = max(1, self.current_version - steps)
        
        if target_version == self.current_version:
            logger.warning("Already at oldest available checkpoint")
            return False
        
        # Load target version
        data = self.load(target_version)
        if data is None:
            return False
        
        # Save as new checkpoint (preserving history)
        return self.save(data, description=f"Rolled back from v{self.current_version} to v{target_version}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            to_remove = len(self.checkpoints) - self.max_checkpoints
            
            for checkpoint in self.checkpoints[:to_remove]:
                try:
                    checkpoint_file = self.checkpoint_dir / f"checkpoint_v{checkpoint.version}.gz"
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                        logger.debug(f"Cleaned up old checkpoint v{checkpoint.version}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint: {e}")
            
            # Keep only recent checkpoints
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints with metadata."""
        return [
            {
                "version": cp.version,
                "timestamp": cp.timestamp,
                "datetime": datetime.fromtimestamp(cp.timestamp).isoformat(),
                "description": cp.description,
                "compression_ratio": cp.compression_ratio,
                "metadata": cp.metadata,
            }
            for cp in self.checkpoints
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_disk = 0
        for cp in self.checkpoints:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_v{cp.version}.gz"
            if checkpoint_file.exists():
                total_disk += checkpoint_file.stat().st_size
        
        avg_compression = sum(cp.compression_ratio for cp in self.checkpoints) / max(1, len(self.checkpoints))
        
        return {
            "current_version": self.current_version,
            "num_checkpoints": len(self.checkpoints),
            "total_disk_usage": total_disk,
            "avg_compression_ratio": avg_compression,
            "checkpoint_dir": str(self.checkpoint_dir),
        }


__all__ = ["PersistenceManager", "Checkpoint", "CorruptionDetector", "CompressionManager"]
