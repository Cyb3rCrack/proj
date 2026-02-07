"""
Integration Example: 4-Layer Storage with ACE Ingestion

This shows how to integrate the 4-layer storage architecture
with ACE's existing ingestion pipeline.

Typical flow:
1. Ingestion module fetches content (web.py, documents.py)
2. Extraction module parses it (parsing.py, concepts.py)
3. 4-layer pipeline stores it (storage_layers.py)
4. Beliefs/claims systems reference it (claims.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from logging import getLogger

# These would be imported from ACE modules
# from Zypherus.extraction import parse_document
# from Zypherus.ingestion import fetch_url
# from Zypherus.beliefs import ClaimStore

from .storage_backend import (
    StorageLayerOrchestrator,
    RawCapture,
    Section,
)
from .ingestion_pipeline import ProgressiveIngestionPipeline

logger = getLogger(__name__)


class ACEStorageIntegration:
    """
    Integration bridge between ACE ingestion and 4-layer storage.
    
    Handles the conversion from ACE's native formats to the 4-layer model.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.orchestrator = StorageLayerOrchestrator(storage_path)
        self.pipeline = ProgressiveIngestionPipeline(self.orchestrator)
    
    def ingest_webpage(
        self,
        url: str,
        html_content: str,
        cleaned_text: str,
        extracted_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a webpage through the 4-layer pipeline.
        
        This is the main entry point for web content.
        
        Args:
            url: Source URL
            html_content: Raw HTML
            cleaned_text: Pre-cleaned/processed text
            extracted_metadata: Any metadata from extraction
        
        Returns:
            Dict with ingestion results
        """
        logger.info(f"Ingesting webpage: {url}")
        
        metadata = extracted_metadata or {}
        metadata.update({
            "source": "web_ingestion",
            "ingestion_time": __import__("time").time(),
        })
        
        # Stage 1: Capture
        capture = self.pipeline.stage_1_capture(
            source_url=url,
            raw_html=html_content,
            cleaned_text=cleaned_text,
            metadata=metadata,
        )
        
        if not capture:
            logger.info(f"Content already captured: {url}")
            return {
                "status": "duplicate",
                "url": url,
            }
        
        # Extract structure from cleaned text
        # In production, this would use ACE's extraction module
        sections = self._extract_sections_from_text(cleaned_text)
        
        # Stage 2-5: Full pipeline
        result = self.pipeline.ingest_page_full_pipeline(
            source_url=url,
            raw_html=html_content,
            cleaned_text=cleaned_text,
            sections=sections,
            metadata=metadata,
        )
        
        return result
    
    def ingest_document(
        self,
        doc_path: Path,
        doc_type: str,
        extracted_content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ingest a document (PDF, Markdown, etc.) through the 4-layer pipeline.
        
        Args:
            doc_path: Local path to document
            doc_type: Type of document (pdf, markdown, etc.)
            extracted_content: Pre-extracted content dict with:
                - text: Full extracted text
                - sections: List of {'title': str, 'content': str}
                - tables: List of table data
                - metadata: Dict with author, date, etc.
        
        Returns:
            Dict with ingestion results
        """
        logger.info(f"Ingesting document: {doc_path}")
        
        url = f"file://{doc_path.absolute()}"
        html_content = f"<!-- Document: {doc_path.name} -->"
        cleaned_text = extracted_content.get("text", "")
        
        metadata = extracted_content.get("metadata", {})
        metadata.update({
            "source": "document_ingestion",
            "doc_type": doc_type,
            "doc_path": str(doc_path),
        })
        
        # Parse sections
        sections = [
            Section(
                section_id=f"sec_{i}",
                title=sec.get("title", f"Section {i}"),
                level=sec.get("level", 2),
                content=sec.get("content", ""),
            )
            for i, sec in enumerate(extracted_content.get("sections", []))
        ]
        
        # Stage 2-5: Full pipeline
        result = self.pipeline.ingest_page_full_pipeline(
            source_url=url,
            raw_html=html_content,
            cleaned_text=cleaned_text,
            sections=sections,
            metadata=metadata,
        )
        
        return result
    
    def ingest_from_youtube_transcript(
        self,
        video_id: str,
        title: str,
        transcript_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest YouTube video transcript through the 4-layer pipeline.
        
        Args:
            video_id: YouTube video ID
            title: Video title
            transcript_text: Full transcript
            metadata: Additional metadata
        
        Returns:
            Dict with ingestion results
        """
        logger.info(f"Ingesting YouTube transcript: {video_id}")
        
        url = f"https://youtube.com/watch?v={video_id}"
        
        meta = metadata or {}
        meta.update({
            "source": "youtube_transcript",
            "video_id": video_id,
            "video_title": title,
        })
        
        # Parse transcript into sections by topic
        sections = self._extract_sections_from_transcript(
            transcript_text,
            title,
        )
        
        result = self.pipeline.ingest_page_full_pipeline(
            source_url=url,
            raw_html=f"<!-- YouTube: {title} -->",
            cleaned_text=transcript_text,
            sections=sections,
            metadata=meta,
        )
        
        return result
    
    def _extract_sections_from_text(self, text: str) -> List[Section]:
        """
        Extract sections from plain text.
        
        In production, this would use ACE's extraction.parsing module.
        """
        sections = []
        
        # Simple heuristic: split by blank lines
        paragraphs = text.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            # First line might be a heading
            lines = para.strip().split("\n")
            title = lines[0][:50] if lines else f"Section {i}"
            
            section = Section(
                section_id=f"section_{i}",
                title=title,
                level=2,
                content=para,
            )
            sections.append(section)
        
        return sections
    
    def _extract_sections_from_transcript(
        self,
        transcript: str,
        video_title: str,
    ) -> List[Section]:
        """
        Extract sections from YouTube transcript.
        
        Transcripts often have timestamps or speaker changes.
        """
        sections = []
        
        # Simple: create one section, could be enhanced
        section = Section(
            section_id="transcript",
            title=f"Transcript: {video_title}",
            level=1,
            content=transcript,
        )
        sections.append(section)
        
        return sections


# Example usage showing integration with ACE components

def example_integrated_ingestion():
    """
    Example: Complete ingestion workflow with 4-layer storage.
    """
    
    # Initialize integration
    integration = ACEStorageIntegration(Path("./storage"))
    
    # Example 1: Ingest a webpage
    print("\n=== Example 1: Ingest Webpage ===")
    
    result = integration.ingest_webpage(
        url="https://example.com/guide",
        html_content="<html><body><h1>Guide</h1>...</body></html>",
        cleaned_text="""
        Introduction to the system.
        
        The system has three components.
        
        Component 1: Storage Layer
        Handles data persistence.
        
        Component 2: Query Layer
        Handles data retrieval.
        """,
        extracted_metadata={
            "author": "docs team",
            "version": "1.0",
        },
    )
    
    print(f"Webpage ingestion result: {result}")
    
    # Example 2: Ingest a document
    print("\n=== Example 2: Ingest Document ===")
    
    doc_result = integration.ingest_document(
        doc_path=Path("./docs/guide.md"),
        doc_type="markdown",
        extracted_content={
            "text": "Full extracted text here...",
            "sections": [
                {"title": "Overview", "level": 2, "content": "Overview text..."},
                {"title": "API Ref", "level": 2, "content": "API details..."},
            ],
            "tables": [],
            "metadata": {"author": "Jane Doe", "date": "2026-02-05"},
        },
    )
    
    print(f"Document ingestion result: {doc_result}")
    
    # Example 3: Ingest YouTube transcript
    print("\n=== Example 3: Ingest YouTube Transcript ===")
    
    yt_result = integration.ingest_from_youtube_transcript(
        video_id="dQw4w9WgXcQ",
        title="How to Build Storage Systems",
        transcript_text="""
        [00:00] Hello everyone, today we're talking about storage systems.
        [00:30] The key principle is separation of concerns.
        [01:00] We have four layers...
        """,
    )
    
    print(f"YouTube ingestion result: {yt_result}")
    
    # Show final storage stats
    print("\n=== Storage Statistics ===")
    integration.orchestrator.print_stats()


def example_ace_ingestion_hook():
    """
    Example: How to hook into ACE's existing ingestion pipeline.
    
    This would be added to ace/ingestion/web.py or similar.
    """
    
    # At the end of ACE's web ingestion:
    
    # Before:
    # 1. Fetch URL (web.py)
    # 2. Parse content (parsing.py)
    # 3. Extract concepts (concepts.py)
    # 4. Store in claims (claims.py)
    
    # After: Add 4-layer storage
    
    # Pseudo-code:
    
    # integration = ACEStorageIntegration(Path("./storage"))
    
    # result = integration.ingest_webpage(
    #     url=fetched_url,
    #     html_content=html,
    #     cleaned_text=cleaned_text,
    #     extracted_metadata=metadata,
    # )
    
    # Then claim extraction can reference the stored chunks
    # instead of duplicating data


def example_migration_preservation():
    """
    Example: Preserving existing claims while migrating to 4-layer.
    
    This allows gradual transition without data loss.
    """
    
    from .migration import ExistingMemoryAdapter, MemoryLayerBridge
    
    # Step 1: Migrate existing data
    print("Step 1: Migrating existing data...")
    
    orch = StorageLayerOrchestrator(Path("./storage"))
    
    adapter = ExistingMemoryAdapter(
        orch,
        memory_paths={
            "claims": Path("claims.json"),
            "definitions": Path("definitions.json"),
            "memory": Path("memory.json"),
        }
    )
    
    migration_results = adapter.run_full_migration()
    print(f"Migration: {migration_results}")
    
    # Step 2: New ingestions use 4-layer
    print("\\nStep 2: New ingestions using 4-layer...")
    
    integration = ACEStorageIntegration(Path("./storage"))
    
    for url in ["https://new-source-1.com", "https://new-source-2.com"]:
        # Simulate fetching and ingesting new content
        result = integration.ingest_webpage(
            url=url,
            html_content="<html>...</html>",
            cleaned_text="New content...",
        )
        print(f"Ingested {url}: {result['chunks_stored']} chunks")
    
    # Step 3: Export for compatibility
    print("\\nStep 3: Exporting for backward compatibility...")
    
    bridge = MemoryLayerBridge(orch)
    
    bridge.export_learned_facts_to_claims(Path("claims_v2.json"))
    bridge.export_semantic_chunks_to_definitions(Path("definitions_v2.json"))
    
    print("Exported: claims_v2.json, definitions_v2.json")
    
    # Step 4: Verify integrity
    print("\\nStep 4: Verifying...")
    
    orch.print_stats()


if __name__ == "__main__":
    print("4-LAYER STORAGE INTEGRATION EXAMPLES")
    print("=" * 60)
    
    example_integrated_ingestion()
    print("\n" + "=" * 60)
    print("\nFor migration example, see example_migration_preservation()")
