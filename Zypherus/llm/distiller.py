"""
Extract and distill knowledge from ingested text.

Features:
- Parallel extraction (facts, summary, entities)
- Caching with LRU eviction and TTL
- Per-task timeout support
- Telemetry counters
- Async support via DistillerAsync
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from .renderer import LLMRenderer
from ..utils.text import get_nlp

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL support."""
    
    def __init__(self, value: Any, ttl_seconds: float):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


class Distiller:
    """Extract facts, summary, and entities from text using LLM with caching."""
    
    def __init__(self, llm: Optional[LLMRenderer] = None):
        """
        Initialize distiller.
        
        Args:
            llm: LLM renderer (creates new if not provided)
        """
        self.llm = llm or LLMRenderer()
        
        # Environment configuration
        self.max_facts = int(os.getenv("ACE_DISTILLER_MAX_FACTS", "10"))
        self.fact_char_limit = int(os.getenv("ACE_DISTILLER_FACT_CHAR_LIMIT", "500"))
        self.summary_char_limit = int(os.getenv("ACE_DISTILLER_SUMMARY_CHAR_LIMIT", "300"))
        self.max_entities = int(os.getenv("ACE_DISTILLER_MAX_ENTITIES", "20"))
        self.workers = int(os.getenv("ACE_DISTILLER_WORKERS", "3"))
        self.cache_ttl_s = float(os.getenv("ACE_CACHE_TTL_S", "3600"))
        
        # Default prompts
        self.fact_extraction_prompt = os.getenv(
            "ACE_PROMPT_FACT_EXTRACTION",
            "Extract up to {max_facts} key facts from this text as bullet points:\n{text}\n\nFacts:"
        )
        self.summary_prompt = os.getenv(
            "ACE_PROMPT_SUMMARY",
            "Summarize this text in {max_chars} characters:\n{text}\n\nSummary:"
        )
        self.entity_extraction_prompt = os.getenv(
            "ACE_PROMPT_ENTITY_EXTRACTION",
            "Extract named entities (person, org, location, etc) from:\n{text}\n\nEntities:"
        )
        
        # Caching (LRU with TTL)
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.max_cache_size = int(os.getenv("ACE_CACHE_MAX_ENTRIES", "100"))
        
        # Telemetry counters (mixed int/float types)
        self.metrics: Dict[str, Union[int, float]] = {
            "extract_facts_count": 0,
            "extract_facts_errors": 0,
            "extract_summary_count": 0,
            "extract_summary_errors": 0,
            "extract_entities_count": 0,
            "extract_entities_errors": 0,
            "spacy_fallback_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "timeout_errors": 0,
            "total_latency_ms": 0.0,
            "total_operations": 0,
        }
        self.metrics_lock = __import__('threading').Lock()
        
        logger.info(
            f"Distiller initialized: max_facts={self.max_facts}, "
            f"workers={self.workers}, cache_ttl={self.cache_ttl_s}s"
        )
    
    def _get_cache_key(self, text: str, method: str) -> str:
        """Generate cache key from text and method."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{method}:{text_hash}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                with self.metrics_lock:
                    self.metrics["cache_hits"] += 1
                return entry.value
            else:
                del self.cache[key]
        
        with self.metrics_lock:
            self.metrics["cache_misses"] += 1
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache with TTL and LRU eviction."""
        self.cache[key] = CacheEntry(value, self.cache_ttl_s)
        
        # LRU eviction
        if len(self.cache) > self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            with self.metrics_lock:
                self.metrics["cache_evictions"] += 1
    
    def _record_metric(self, metric_name: str, increment: int = 1, latency_ms: float = 0) -> None:
        """Update telemetry counter safely."""
        with self.metrics_lock:
            if metric_name in self.metrics:
                self.metrics[metric_name] += increment
            if latency_ms > 0:
                self.metrics["total_latency_ms"] += latency_ms
                self.metrics["total_operations"] += 1
    
    def _check_deadline(self, deadline_s: Optional[float]) -> None:
        """Raise TimeoutError if deadline exceeded."""
        if deadline_s is not None and time.time() > deadline_s:
            raise TimeoutError("Task deadline exceeded")
    
    def extract_facts(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> List[str]:
        """
        Extract key facts from text.
        
        Args:
            text: Text to analyze
            deadline_s: Unix timestamp deadline for completion
            
        Returns:
            List of fact strings
            
        Raises:
            TimeoutError: If deadline exceeded
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, "extract_facts")
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._record_metric("extract_facts_count", latency_ms=(time.time() - start_time) * 1000)
                return cached
            
            self._check_deadline(deadline_s)
            
            # Generate facts via LLM
            prompt = self.fact_extraction_prompt.format(
                max_facts=self.max_facts,
                text=text[:2000]  # Limit text to avoid token limits
            )
            
            response = self.llm.generate(
                prompt,
                deadline_s=deadline_s,
                max_tokens=int(os.getenv("ZYPHERUS_MAX_TOKENS_EXTRACTION", "500"))
            )
            
            # Parse facts
            facts = [
                line.strip("- ").strip()
                for line in response.split("\n")
                if line.strip().startswith("-") or (line.strip() and not line.startswith("#"))
            ][:self.max_facts]
            
            # Filter empty and store in cache
            facts = [f for f in facts if len(f) > 0 and len(f) <= self.fact_char_limit]
            self._set_cached(cache_key, facts)
            
            self._record_metric("extract_facts_count", latency_ms=(time.time() - start_time) * 1000)
            logger.info(f"Extracted {len(facts)} facts from text")
            return facts
            
        except TimeoutError as e:
            self._record_metric("extract_facts_errors")
            self._record_metric("timeout_errors")
            logger.warning(f"Fact extraction timeout: {e}")
            raise
        except Exception as e:
            self._record_metric("extract_facts_errors")
            logger.error(f"Fact extraction error: {e}", exc_info=True)
            return []
    
    def extract_summary(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> str:
        """
        Extract summary from text.
        
        Args:
            text: Text to summarize
            deadline_s: Unix timestamp deadline for completion
            
        Returns:
            Summary string
            
        Raises:
            TimeoutError: If deadline exceeded
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, "extract_summary")
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._record_metric("extract_summary_count", latency_ms=(time.time() - start_time) * 1000)
                return cached
            
            self._check_deadline(deadline_s)
            
            # Generate summary via LLM
            prompt = self.summary_prompt.format(
                max_chars=self.summary_char_limit,
                text=text[:2000]
            )
            
            response = self.llm.generate(
                prompt,
                deadline_s=deadline_s,
                max_tokens=int(os.getenv("ZYPHERUS_MAX_TOKENS_EXTRACTION", "500"))
            )
            
            summary = response.strip()[:self.summary_char_limit]
            self._set_cached(cache_key, summary)
            
            self._record_metric("extract_summary_count", latency_ms=(time.time() - start_time) * 1000)
            logger.info(f"Extracted summary ({len(summary)} chars)")
            return summary
            
        except TimeoutError as e:
            self._record_metric("extract_summary_errors")
            self._record_metric("timeout_errors")
            logger.warning(f"Summary extraction timeout: {e}")
            raise
        except Exception as e:
            self._record_metric("extract_summary_errors")
            logger.error(f"Summary extraction error: {e}", exc_info=True)
            return ""
    
    def extract_entities(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> List[Dict[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            deadline_s: Unix timestamp deadline for completion
            
        Returns:
            List of {entity, label} dicts
            
        Raises:
            TimeoutError: If deadline exceeded
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, "extract_entities")
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._record_metric("extract_entities_count", latency_ms=(time.time() - start_time) * 1000)
                return cached
            
            self._check_deadline(deadline_s)
            
            # Try spaCy first
            try:
                nlp = get_nlp()
                if nlp is not None:
                    doc = nlp(text[:2000])
                    entities = [
                        {"entity": ent.text, "label": ent.label_}
                        for ent in doc.ents
                    ][:self.max_entities]
                    
                    if entities:
                        self._set_cached(cache_key, entities)
                        self._record_metric("extract_entities_count", latency_ms=(time.time() - start_time) * 1000)
                        logger.info(f"Extracted {len(entities)} entities via spaCy")
                        return entities
            except Exception as e:
                logger.debug(f"spaCy fallback triggered: {e}")
                self._record_metric("spacy_fallback_count")
            
            # Fallback to LLM
            prompt = self.entity_extraction_prompt.format(text=text[:2000])
            response = self.llm.generate(
                prompt,
                deadline_s=deadline_s,
                max_tokens=int(os.getenv("ZYPHERUS_MAX_TOKENS_EXTRACTION", "500"))
            )
            
            # Parse entities from LLM response
            entities = []
            try:
                # Try JSON format first
                if "{" in response:
                    json_str = response[response.find("{"):response.rfind("}") + 1]
                    parsed = json.loads(json_str)
                    entities = parsed.get("entities", [])[:self.max_entities]
            except:
                # Parse line by line
                for line in response.split("\n"):
                    if ":" in line:
                        parts = line.split(":", 1)
                        entities.append({
                            "entity": parts[0].strip(),
                            "label": parts[1].strip() if len(parts) > 1 else "UNKNOWN"
                        })
            
            entities = entities[:self.max_entities]
            self._set_cached(cache_key, entities)
            
            self._record_metric("extract_entities_count", latency_ms=(time.time() - start_time) * 1000)
            logger.info(f"Extracted {len(entities)} entities via LLM")
            return entities
            
        except TimeoutError as e:
            self._record_metric("extract_entities_errors")
            self._record_metric("timeout_errors")
            logger.warning(f"Entity extraction timeout: {e}")
            raise
        except Exception as e:
            self._record_metric("extract_entities_errors")
            logger.error(f"Entity extraction error: {e}", exc_info=True)
            return []
    
    def extract_all(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract facts, summary, and entities in parallel.
        
        Args:
            text: Text to analyze
            deadline_s: Unix timestamp deadline for completion
            
        Returns:
            Dict with keys: facts, summary, entities
            
        Raises:
            TimeoutError: If deadline exceeded
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.extract_facts, text, deadline_s): "facts",
                executor.submit(self.extract_summary, text, deadline_s): "summary",
                executor.submit(self.extract_entities, text, deadline_s): "entities",
            }
            
            for future in as_completed(futures, timeout=None):
                key = futures[future]
                try:
                    results[key] = future.result()
                except TimeoutError:
                    logger.warning(f"Timeout in {key} extraction")
                    results[key] = [] if key != "summary" else ""
                except Exception as e:
                    logger.error(f"Error in {key} extraction: {e}")
                    results[key] = [] if key != "summary" else ""
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry counters."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
            if metrics["total_operations"] > 0:
                metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_operations"]
            return metrics
    
    def reset_metrics(self) -> None:
        """Reset all telemetry counters."""
        with self.metrics_lock:
            for key in self.metrics:
                if isinstance(self.metrics[key], int):
                    self.metrics[key] = 0
                elif isinstance(self.metrics[key], float):
                    self.metrics[key] = 0.0


class DistillerAsync:
    """Async version of Distiller using asyncio and httpx."""
    
    def __init__(self, llm: Optional[LLMRenderer] = None):
        """
        Initialize async distiller.
        
        Args:
            llm: LLM renderer (creates new if not provided)
        """
        self.llm = llm or LLMRenderer()
        self.sync_distiller = Distiller(llm)
    
    async def extract_facts(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> List[str]:
        """Extract facts asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_distiller.extract_facts(text, deadline_s)
        )
    
    async def extract_summary(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> str:
        """Extract summary asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_distiller.extract_summary(text, deadline_s)
        )
    
    async def extract_entities(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> List[Dict[str, str]]:
        """Extract entities asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_distiller.extract_entities(text, deadline_s)
        )
    
    async def extract_all(
        self,
        text: str,
        deadline_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract facts, summary, and entities in parallel using asyncio.
        
        Args:
            text: Text to analyze
            deadline_s: Unix timestamp deadline for completion
            
        Returns:
            Dict with keys: facts, summary, entities
        """
        try:
            facts, summary, entities = await asyncio.gather(
                self.extract_facts(text, deadline_s),
                self.extract_summary(text, deadline_s),
                self.extract_entities(text, deadline_s),
                return_exceptions=True
            )
            
            return {
                "facts": facts if not isinstance(facts, Exception) else [],
                "summary": summary if not isinstance(summary, Exception) else "",
                "entities": entities if not isinstance(entities, Exception) else [],
            }
        except Exception as e:
            logger.error(f"Error in async extract_all: {e}", exc_info=True)
            return {"facts": [], "summary": "", "entities": []}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry counters from underlying sync distiller."""
        return self.sync_distiller.get_metrics()
