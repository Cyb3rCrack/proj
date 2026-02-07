"""YouTube transcript ingestion for ACE.

Mental model: Videos are multi-modal documents (speech, text, metadata, visuals).
Processing: Transcript → Filter opinions → Segment into atomic knowledge → Ingest.

This module handles transcripts only (80-90% of knowledge value), not full video processing.
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Classification of transcript content."""
    DECLARATIVE_FACT = "declarative_fact"  # Can be ingested
    DEFINITION = "definition"              # Can be ingested
    PROCESS = "process"                    # Can be ingested
    EXAMPLE = "example"                    # Can be ingested
    OPINION = "opinion"                    # Must filter
    SPECULATION = "speculation"            # Must filter
    FILLER = "filler"                      # Must filter
    PROMOTION = "promotion"                # Must filter (sponsor reads, ads)
    UNCLEAR = "unclear"                    # Must filter


@dataclass
class TranscriptSegment:
    """A validated, filtered segment ready for ingestion."""
    text: str
    start_time: float
    end_time: float
    content_type: ContentType
    is_valid: bool
    confidence: float
    

@dataclass
class KnowledgeChunk:
    """Atomic knowledge unit extracted from video."""
    source_video: str
    source_channel: str
    timestamp: str
    topic: str
    content: str
    content_type: ContentType


class TranscriptValidator:
    """Validates transcript content before ingestion."""
    
    # Patterns that indicate opinion/speculation (STRICT - only clear opinions)
    OPINION_PATTERNS = [
        r"\bi\s+(?:think|believe|feel|consider|suppose)\b",
        r"\bin\s+my\s+(?:opinion|view|experience|experience)\b",
        r"\bimho\b",
        r"I'm\s+(?:not\s+)?sure",
    ]
    
    # Patterns that indicate filler/non-content (STRICT - only clear filler)
    FILLER_PATTERNS = [
        r"^(?:um|uh|like|you\s+know|so)\b",
        r"^(?:thanks for watching|don't forget to like|subscribe)",
        r"^(?:welcome|hello|hi|good morning|good afternoon)",
        r"^(?:thanks to|this video is brought to you by|sponsor)",
    ]
    
    @staticmethod
    def classify_sentence(text: str) -> Tuple[ContentType, float]:
        """Classify sentence and return type + confidence.
        
        Args:
            text: Single sentence or short phrase
            
        Returns:
            (ContentType, confidence_0_to_1)
        """
        text_lower = text.lower().strip()
        
        # Quick length check - very short (<5 words) is likely filler
        word_count = len(text_lower.split())
        if word_count < 3:
            return ContentType.FILLER, 0.7
        
        # Check for STRONG opinion markers
        for pattern in TranscriptValidator.OPINION_PATTERNS:
            if re.search(pattern, text_lower):
                return ContentType.OPINION, 0.9
        
        # Check for filler markers
        for pattern in TranscriptValidator.FILLER_PATTERNS:
            if re.search(pattern, text_lower):
                return ContentType.FILLER, 0.85
        
        # DEFAULT: If not rejected, assume it's content
        # We use a permissive approach - only reject if we KNOW it's bad
        # This is safer than rejecting too aggressively
        
        # Weak rejection signals
        weak_reject_keywords = ["i think", "maybe", "probably", "might be", "could be"]
        has_weak_reject = any(kw in text_lower for kw in weak_reject_keywords)
        
        if has_weak_reject:
            # Lower confidence but still allow
            return ContentType.DECLARATIVE_FACT, 0.45
        
        # Strong acceptance if contains known knowledge indicators
        knowledge_keywords = [
            "is", "are", "was", "were", "uses", "means", "defined", 
            "called", "refers", "algorithm", "method", "function", 
            "parameter", "returns", "creates", "stores", "represents",
            "allows", "enables", "prevents", "supports", "requires",
        ]
        
        has_knowledge = any(f" {kw} " in f" {text_lower} " for kw in knowledge_keywords)
        
        if has_knowledge:
            return ContentType.DECLARATIVE_FACT, 0.75
        
        # If none of above, still default to DECLARATIVE_FACT with medium confidence
        # Rather than filter too aggressively
        return ContentType.DECLARATIVE_FACT, 0.55
    
    @staticmethod
    def is_ingestible(text: str, confidence_threshold: float = 0.6) -> Tuple[bool, ContentType, float]:
        """Check if a segment should be ingested.
        
        Returns:
            (should_ingest, content_type, confidence)
        """
        content_type, confidence = TranscriptValidator.classify_sentence(text)
        
        # Only filter if we're confident it's BAD
        bad_types = {ContentType.OPINION, ContentType.FILLER, ContentType.UNCLEAR}
        
        should_ingest = content_type not in bad_types or confidence < 0.7
        
        return should_ingest, content_type, confidence


class TranscriptSegmenter:
    """Splits transcript into atomic knowledge chunks."""

    def __init__(self, min_sentences: int = 3, max_sentences: int = 5):
        self.min_sentences = int(min_sentences)
        self.max_sentences = int(max_sentences)
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split transcript into sentences, preserving some context.
        
        Handles abbreviations and edge cases.
        """
        # Common abbreviations to avoid false splits
        abbreviations = r"(?:Dr|Mr|Mrs|Ms|Prof|St|vs|i\.e|e\.g|etc|Ltd|Inc|Corp)"
        
        # Replace abbreviation periods with placeholder
        text = re.sub(rf"({abbreviations})\.", r"\1<DOT>", text)
        
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        # Restore abbreviation periods
        sentences = [s.replace("<DOT>", ".") for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def segment_by_topic(self, text: str) -> List[str]:
        """Group sentences into logical topic chunks (tunable size)."""
        sentences = TranscriptSegmenter.split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Topic boundary: Major topic changes start with transition phrases
            is_topic_end = len(current_chunk) >= self.min_sentences and any(
                sentence.startswith(phrase) 
                for phrase in ["So", "Now", "Next", "Then", "Another", "Also", "In", "The"]
            )
            
            if is_topic_end or len(current_chunk) >= self.max_sentences:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add remainder
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return [c.strip() for c in chunks if len(c.strip()) > 15]


class YouTubeTranscriptProcessor:
    """End-to-end transcript processing: fetch → filter → chunk → validate."""
    
    def __init__(self, confidence_threshold: float = 0.6, fast_mode: Optional[bool] = None):
        self.validator = TranscriptValidator()
        self.fast_mode = fast_mode if fast_mode is not None else os.getenv("ZYPHERUS_INGEST_FAST", "true").lower() in ("1", "true", "yes")
        self.fast_profile = os.getenv("ZYPHERUS_INGEST_PROFILE", "safe").lower()

        min_sentences = 3
        max_sentences = 5
        if self.fast_mode:
            if self.fast_profile == "max":
                min_sentences = 8
                max_sentences = 12
                confidence_threshold = min(confidence_threshold, 0.45)
            else:
                min_sentences = 5
                max_sentences = 8
                confidence_threshold = min(confidence_threshold, 0.5)

        self.segmenter = TranscriptSegmenter(min_sentences=min_sentences, max_sentences=max_sentences)
        self.confidence_threshold = confidence_threshold
    
    def process_transcript(
        self,
        transcript: str,
        video_id: str,
        channel: str,
        title: str,
    ) -> List[KnowledgeChunk]:
        """Process raw transcript into ingestible knowledge chunks.
        
        Args:
            transcript: Full video transcript text
            video_id: YouTube video ID
            channel: Channel name
            title: Video title
            
        Returns:
            List of validated, filtered knowledge chunks
        """
        # Step 1: Segment into atomic chunks
        chunks = self.segmenter.segment_by_topic(transcript)
        
        knowledge_chunks = []
        
        for chunk in chunks:
            # Step 2: Validate each chunk
            if self.fast_mode and self.fast_profile == "max":
                is_valid = True
                content_type = ContentType.DECLARATIVE_FACT
                confidence = 0.5
            else:
                is_valid, content_type, confidence = self.validator.is_ingestible(
                    chunk,
                    confidence_threshold=self.confidence_threshold
                )
                
                if not is_valid:
                    continue  # Skip opinions, filler, speculation
            
            # Step 3: Create knowledge chunk with full metadata
            kc = KnowledgeChunk(
                source_video=f"youtube:{video_id}:{title[:30]}",
                source_channel=channel,
                timestamp="0:00",  # Would be set from actual transcript with times
                topic=self._extract_topic(chunk),
                content=chunk,
                content_type=content_type,
            )
            
            knowledge_chunks.append(kc)
        
        return knowledge_chunks
    
    @staticmethod
    def _extract_topic(text: str) -> str:
        """Extract likely topic from chunk (first few significant words)."""
        # Remove common words
        stop_words = {"is", "are", "the", "a", "an", "and", "or", "but", "if", "when"}
        
        words = [w for w in text.split()[:5] if w.lower() not in stop_words and len(w) > 2]
        
        return "_".join(words[:3]).lower()
    
    def process_transcript_with_stats(self, transcript: str, video_id: str, 
                                      channel: str, title: str) -> Dict:
        """Process and return detailed statistics."""
        chunks = self.process_transcript(transcript, video_id, channel, title)
        
        # Analyze filtering results
        all_segments = self.segmenter.segment_by_topic(transcript)
        
        filtered_count = len(all_segments) - len(chunks)
        
        return {
            "total_segments": len(all_segments),
            "ingested_chunks": len(chunks),
            "filtered_out": filtered_count,
            "ingestion_rate": len(chunks) / len(all_segments) if all_segments else 0,
            "chunks": chunks,
        }


# Example usage
if __name__ == "__main__":
    # Test with sample transcript
    sample_transcript = """
    In Python, a list is a collection of items stored in a single variable.
    Lists are created using square brackets.
    I think Python is better than Java, honestly.
    You know, anyway, moving on to dictionaries.
    A dictionary is an unordered collection of key-value pairs.
    In my experience, dictionaries are super useful.
    When you need to look up values by name, you should use a dictionary.
    The syntax for creating a dictionary uses curly braces with keys and values.
    So, um, thanks for watching this tutorial.
    """
    
    processor = YouTubeTranscriptProcessor(confidence_threshold=0.6)
    stats = processor.process_transcript_with_stats(
        sample_transcript,
        "abc123",
        "PythonTutorials",
        "Python Lists and Dicts"
    )
    
    print(f"Total segments: {stats['total_segments']}")
    print(f"Ingested chunks: {stats['ingested_chunks']}")
    print(f"Filtered out: {stats['filtered_out']}")
    print(f"Ingestion rate: {stats['ingestion_rate']:.1%}\n")
    
    print("Ingested knowledge chunks:")
    for i, chunk in enumerate(stats['chunks'], 1):
        print(f"\n{i}. Topic: {chunk.topic}")
        print(f"   Type: {chunk.content_type.value}")
        print(f"   Content: {chunk.content[:80]}...")
