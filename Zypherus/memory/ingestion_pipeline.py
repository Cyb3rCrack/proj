"""
Progressive Ingestion Pipeline for 4-Layer Storage.

Flow: Capture → Structure → Score → Embed → Synthesize
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from logging import getLogger

from .storage_backend import (
    RawCapture,
    StructuredPage,
    Section,
    ChunkedContent,
    LearnedFact,
    StorageLayerOrchestrator,
)

logger = getLogger(__name__)


class ImportanceScorer:
    """Score content importance for semantic storage."""
    
    @staticmethod
    def score_content_type(content_type: str) -> float:
        """Base score for content type."""
        scoring = {
            "definition": 0.95,
            "specification": 0.90,
            "algorithm": 0.88,
            "rule": 0.85,
            "concept": 0.80,
            "explanation": 0.65,
            "example": 0.45,
            "implementation": 0.70,
            "tutorial": 0.50,
            "filler": 0.1,
            "navigation": 0.05,
            "legal": 0.02,
        }
        return scoring.get(content_type, 0.3)
    
    @staticmethod
    def compute_information_density(text: str) -> float:
        """
        Compute information density of text (0.0 - 1.0).
        Higher = more technical/specific content.
        """
        if not text:
            return 0.0
        
        # Count technical indicators
        lines = text.split("\n")
        
        # Metrics
        code_lines = sum(1 for line in lines if any(c in line for c in ["(", ")", "{", "}"]))
        tech_words = sum(
            1 for line in lines
            for word in ["parameter", "function", "return", "argument", "variable", "object"]
            if word.lower() in line.lower()
        )
        
        # Density = (technical features) / (total words)
        words = len(text.split())
        if words == 0:
            return 0.0
        
        technical_count = code_lines + tech_words
        density = min(technical_count / (words / 50), 1.0)  # Normalize
        
        return density
    
    @staticmethod
    def compute_novelty(text: str, existing_texts: List[str]) -> float:
        """
        Compute novelty of text vs existing content (0.0 - 1.0).
        1.0 = completely novel
        0.0 = duplicated content
        """
        if not existing_texts:
            return 1.0
        
        # Compute similarity to closest existing text
        text_words = set(text.lower().split())
        
        max_similarity = 0.0
        for existing in existing_texts:
            existing_words = set(existing.lower().split())
            if not existing_words or not text_words:
                continue
            
            overlap = len(text_words & existing_words)
            union = len(text_words | existing_words)
            jaccard = overlap / union if union > 0 else 0
            
            max_similarity = max(max_similarity, jaccard)
        
        novelty = 1.0 - max_similarity
        return novelty
    
    @staticmethod
    def count_cross_references(chunk_id: str, all_chunks: Dict[str, ChunkedContent]) -> int:
        """Count how many other chunks reference this one."""
        # In production, this would parse links in chunk text
        # For now, return 0 (would be populated during synthesis)
        return 0


class ProgressiveIngestionPipeline:
    """
    Progressive 4-layer ingestion pipeline.
    
    Stage 1: Capture raw content
    Stage 2: Extract structure
    Stage 3: Score importance
    Stage 4: Create semantic chunks with embeddings
    Stage 5: Synthesize learned facts
    """
    
    def __init__(self, orchestrator: StorageLayerOrchestrator):
        self.orchestrator = orchestrator
        self.scorer = ImportanceScorer()
    
    def stage_1_capture(
        self,
        source_url: str,
        raw_html: str,
        cleaned_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RawCapture]:
        """
        STAGE 1: Store raw content.
        
        Returns RawCapture if new, None if duplicate.
        """
        if metadata is None:
            metadata = {}
        
        content_hash = RawCapture.compute_hash(raw_html)
        
        capture = RawCapture(
            source_url=source_url,
            content_hash=content_hash,
            raw_html=raw_html,
            cleaned_text=cleaned_text,
            metadata=metadata,
            ingestion_version=time.strftime("%Y-%m-%d"),
        )
        
        is_new = self.orchestrator.raw.store_capture(capture)
        return capture if is_new else None
    
    def stage_2_structure(
        self,
        capture: RawCapture,
        sections: Optional[List[Section]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        code_blocks: Optional[List[Dict[str, str]]] = None,
        internal_links: Optional[List[Dict[str, str]]] = None,
        version_marker: str = "",
    ) -> str:
        """
        STAGE 2: Extract and store structure.
        
        Returns page_id for reference in later stages.
        """
        page = StructuredPage(
            source_url=capture.source_url,
            raw_content_hash=capture.content_hash,
            sections=sections or [],
            tables=tables or [],
            code_blocks=code_blocks or [],
            internal_links=internal_links or [],
            version_marker=version_marker,
        )
        
        page_id = self.orchestrator.structural.store_page(page)
        logger.info(f"Structured extraction complete: {page_id}")
        
        return page_id
    
    def stage_3_score(
        self,
        content_type: str,
        text: str,
        existing_semantic_texts: Optional[List[str]] = None,
        user_query_relevant: bool = False,
    ) -> float:
        """
        STAGE 3: Score importance before semantic storage.
        
        Combines:
        - Content type value
        - Information density
        - Novelty vs existing
        - Query relevance
        """
        if existing_semantic_texts is None:
            existing_semantic_texts = []
        
        # Component scores
        type_score = self.scorer.score_content_type(content_type)
        density_score = self.scorer.compute_information_density(text)
        novelty_score = self.scorer.compute_novelty(text, existing_semantic_texts)
        query_boost = 0.15 if user_query_relevant else 0.0
        
        # Weighted combination
        importance = (
            type_score * 0.5 +
            density_score * 0.2 +
            novelty_score * 0.2 +
            query_boost
        )
        
        return min(importance, 1.0)
    
    def stage_4_semantic(
        self,
        page_id: str,
        content_type: str,
        text: str,
        source_url: str,
        topic_tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        importance_score: Optional[float] = None,
    ) -> Optional[str]:
        """
        STAGE 4: Create semantic chunk with embedding.
        
        Only stores if importance >= threshold.
        Returns chunk_id if stored, None if rejected.
        """
        if topic_tags is None:
            topic_tags = []
        
        # Compute score if not provided
        if importance_score is None:
            importance_score = self.stage_3_score(
                content_type,
                text,
                existing_semantic_texts=[],  # Would need to search semantic layer
            )
        
        chunk_id = hashlib.md5(
            f"{page_id}:{content_type}:{len(text)}".encode()
        ).hexdigest()
        
        chunk = ChunkedContent(
            chunk_id=chunk_id,
            source_url=source_url,
            page_id=page_id,
            content_type=content_type,
            text=text,
            embedding=embedding,
            topic_tags=topic_tags,
            importance_score=importance_score,
            information_density=self.scorer.compute_information_density(text),
        )
        
        stored = self.orchestrator.semantic.store_chunk(chunk)
        return chunk_id if stored else None
    
    def stage_5_synthesize(
        self,
        chunk_id: str,
        chunked_content: ChunkedContent,
        confidence: float = 0.8,
        expires_days: Optional[int] = None,
    ) -> Optional[str]:
        """
        STAGE 5: Synthesize learned fact from semantic chunk.
        
        Transforms high-quality chunks into answer-ready knowledge.
        Returns fact_id if stored, None if not worthy of synthesis.
        """
        # Only synthesize high-confidence, high-importance chunks
        if chunk_id not in self.orchestrator.semantic.chunks:
            logger.warning(f"Chunk not found: {chunk_id}")
            return None
        
        chunk = self.orchestrator.semantic.chunks[chunk_id]
        
        # Synthesis worthy only if high importance
        if chunk.importance_score < 0.7:
            logger.debug(f"Chunk {chunk_id} not synthesized (low importance)")
            return None
        
        # Create learned fact from chunk
        fact_id = f"fact:{hashlib.md5(chunked_content.text[:100].encode()).hexdigest()}"
        
        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 86400)
        
        fact = LearnedFact(
            fact_id=fact_id,
            statement=chunked_content.text,
            sources=[chunk_id],
            confidence=confidence,
            expires_at=expires_at,
            tags=chunked_content.topic_tags,
            metadata={
                "content_type": chunked_content.content_type,
                "source_url": chunked_content.source_url,
            },
        )
        
        fact_id = self.orchestrator.learned.store_fact(fact)
        logger.info(f"Synthesized learned fact: {fact_id}")
        
        return fact_id
    
    def ingest_page_full_pipeline(
        self,
        source_url: str,
        raw_html: str,
        cleaned_text: str,
        sections: List[Section],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline for a page.
        
        Orchestrates all 5 stages and returns summary.
        """
        logger.info(f"Starting full pipeline for: {source_url}")
        
        results = {
            "source_url": source_url,
            "stages_completed": [],
            "page_id": None,
            "chunks_stored": 0,
            "facts_synthesized": 0,
        }
        
        # STAGE 1: Capture
        capture = self.stage_1_capture(
            source_url=source_url,
            raw_html=raw_html,
            cleaned_text=cleaned_text,
            metadata=metadata or {},
        )
        
        if not capture:
            logger.warning(f"Content already captured: {source_url}")
            return results
        
        results["stages_completed"].append("capture")
        
        # STAGE 2: Structure
        page_id = self.stage_2_structure(
            capture=capture,
            sections=sections,
            metadata=metadata,
        )
        results["page_id"] = page_id
        results["stages_completed"].append("structure")
        
        # STAGE 3-4: Score & Semantic for each section
        for section in sections:
            # Score
            importance = self.stage_3_score(
                content_type="explanation",
                text=section.content,
            )
            
            # Semantic
            chunk_id = self.stage_4_semantic(
                page_id=page_id,
                content_type="explanation",
                text=section.content,
                source_url=source_url,
                topic_tags=[section.title.lower()],
                importance_score=importance,
            )
            
            if chunk_id:
                results["chunks_stored"] += 1
            
            # STAGE 5: Synthesize
            if chunk_id:
                chunk = self.orchestrator.semantic.chunks[chunk_id]
                fact_id = self.stage_5_synthesize(
                    chunk_id=chunk_id,
                    chunked_content=chunk,
                    expires_days=30,
                )
                if fact_id:
                    results["facts_synthesized"] += 1
        
        results["stages_completed"].append("score")
        results["stages_completed"].append("semantic")
        results["stages_completed"].append("synthesize")
        
        logger.info(f"Pipeline complete: {results}")
        return results


def create_example_ingestion_flow():
    """Example: Using the progressive pipeline."""
    storage_path = Path("./storage")
    orchestrator = StorageLayerOrchestrator(storage_path)
    pipeline = ProgressiveIngestionPipeline(orchestrator)
    
    # Example page
    sections = [
        Section(
            section_id="intro",
            title="Overview",
            level=2,
            content="This is a definition of the core concept.",
        ),
        Section(
            section_id="api",
            title="API Reference",
            level=2,
            content="def function(param): returns result",
        ),
    ]
    
    result = pipeline.ingest_page_full_pipeline(
        source_url="https://example.com/guide",
        raw_html="<html>...</html>",
        cleaned_text="This is clean text",
        sections=sections,
        metadata={"author": "example", "date": "2026-02-05"},
    )
    
    print(f"Ingestion result: {result}")
    
    # Print storage stats
    orchestrator.print_stats()
