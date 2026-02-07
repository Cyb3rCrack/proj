"""ACE integration for YouTube transcripts.

Handles end-to-end YouTube ingestion:
1. Fetch transcript from channel
2. Validate and filter content
3. Segment into atomic knowledge
4. Ingest into ACE's memory
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
from .youtube_transcript import YouTubeTranscriptProcessor, KnowledgeChunk
from .youtube_curator import YouTubeCurator, CreatorTrust, TrustedChannel, YouTubeIngestionPolicy


class YouTubeIngestionManager:
    """Manages YouTube transcript ingestion for ACE."""
    
    INGESTION_LOG_FILE = "data/ingestion/youtube_ingestions.json"
    
    def __init__(self, ace_instance=None):
        """Initialize with reference to ACE for direct ingestion.
        
        Args:
            ace_instance: ACE core instance (for calling ace.learn())
        """
        self.ace = ace_instance
        self.curator = YouTubeCurator()
        self.curator.add_recommended_channels()
        self.processor = YouTubeTranscriptProcessor(
            confidence_threshold=YouTubeIngestionPolicy.MIN_CONFIDENCE_FOR_INGESTION
        )
        self.ingestion_results = []
    
    def _get_log_file_path(self) -> Path:
        """Get path to youtube ingestions log file."""
        # Try to find project root
        current = Path(__file__).parent
        while current != current.parent:
            if (current / self.INGESTION_LOG_FILE).exists():
                return current / self.INGESTION_LOG_FILE
            if (current / "ace").exists() and (current / "data" / "memory" / "memory.json").exists():
                return current / self.INGESTION_LOG_FILE
            current = current.parent
        # Default to current directory
        return Path(self.INGESTION_LOG_FILE)
    
    def _load_ingestion_log(self) -> Dict:
        """Load existing ingestion log or create new one."""
        log_path = self._get_log_file_path()
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"version": 1, "ingestions": []}
        return {"version": 1, "ingestions": []}
    
    def _save_ingestion_log(self, log_data: Dict) -> None:
        """Save ingestion log to file."""
        log_path = self._get_log_file_path()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save ingestion log: {e}")
    
    def _log_ingestion_to_file(self, report: Dict) -> None:
        """Log ingestion details to youtube_ingestions.json."""
        log_data = self._load_ingestion_log()
        
        # Create ingestion entry
        ingestion_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_id": report.get("video_id"),
            "video_url": report.get("video_url"),
            "video_title": report.get("video_title"),
            "channel": report.get("channel"),
            "chunks_ingested": report.get("chunks_ingested", 0),
            "total_chunks": report.get("total_chunks", 0),
            "ingestion_rate": report.get("ingestion_rate"),
            "status": report.get("status"),
        }
        
        # Add to list
        log_data["ingestions"].append(ingestion_entry)
        
        # Save back to file
        self._save_ingestion_log(log_data)
    
    def ingest_from_transcript(
        self,
        transcript: str,
        video_id: str,
        channel_id: str,
        video_title: str,
        video_url: str = "",
        channel_name: Optional[str] = None,
    ) -> Dict:
        """Ingest a YouTube transcript into ACE.
        
        Args:
            transcript: Full video transcript text
            video_id: YouTube video ID
            channel_id: Channel identifier (from curator)
            video_title: Video title
            video_url: Full YouTube URL (for reference)
            
        Returns:
            Ingestion report with statistics
        """
        # Step 1: Ensure channel is known (or create unvetted entry)
        channel = self.curator.get_channel(channel_id)
        if not channel and YouTubeIngestionPolicy.ALLOW_UNVETTED_CHANNELS:
            display_name = channel_name or channel_id
            channel = TrustedChannel(
                channel_id=channel_id,
                channel_name=display_name,
                owner=display_name,
                description="Unvetted channel (auto-added)",
                expertise=["unspecified"],
                trust_level=CreatorTrust.UNVETTED,
                max_videos_per_month=1000,
                video_quality_criteria=["has_captions"],
            )
            self.curator.add_trusted_channel(channel)

        if not channel:
            return {
                "status": "rejected",
                "reason": f"Channel {channel_id} not in curated list",
                "chunks_ingested": 0,
            }
        
        # Step 2: Check ingestion quota
        if not self.curator.can_ingest_from_channel(channel_id):
            return {
                "status": "rejected",
                "reason": f"Channel {channel_id} exceeded monthly quota",
                "chunks_ingested": 0,
            }
        
        # Step 3: Process transcript
        stats = self.processor.process_transcript_with_stats(
            transcript,
            video_id,
            channel.channel_name,
            video_title,
        )
        
        chunks: List[KnowledgeChunk] = stats["chunks"]
        
        # Step 4: Apply safety policy
        ingested_count = 0
        rejected_reasons = []
        
        for i, chunk in enumerate(chunks):
            # Check policy compliance
            compliant = YouTubeIngestionPolicy.check_policy_compliance(
                channel_trust=channel.trust_level,
                has_captions=True,  # Assumed if we have transcript
                confidence=0.7,  # Would come from actual processing
                auto_reject_found=any(
                    kw in chunk.content.lower() 
                    for kw in YouTubeIngestionPolicy.AUTO_REJECT_KEYWORDS
                ),
            )
            
            if not compliant:
                rejected_reasons.append(f"Chunk {i}: Policy violation")
                continue
            
            # Step 5: Ingest into ACE (if available)
            if self.ace:
                try:
                    source_name = f"youtube:{channel_id}:{video_id}:{chunk.topic}"
                    self.ace.learn(chunk.content, source=source_name)
                    ingested_count += 1
                except Exception as e:
                    rejected_reasons.append(f"Chunk {i}: {str(e)}")
            else:
                # Just count if no Zypherus instance
                ingested_count += 1
        
        # Step 6: Log ingestion
        self.curator.log_ingestion(channel_id, video_id, ingested_count)
        
        # Return report
        report = {
            "status": "success",
            "channel": channel.channel_name,
            "video_id": video_id,
            "video_title": video_title,
            "video_url": video_url,
            "total_segments": stats["total_segments"],
            "total_chunks": len(chunks),
            "chunks_ingested": ingested_count,
            "filtered_out": len(chunks) - ingested_count,
            "policy_rejected": len(rejected_reasons),
            "rejection_reasons": rejected_reasons,
            "ingestion_rate": f"{(ingested_count / len(chunks) * 100):.1f}%" if chunks else "N/A",
        }
        
        # Log to youtube_ingestions.json
        self._log_ingestion_to_file(report)
        
        self.ingestion_results.append(report)
        return report
    
    def dry_run_review(self, transcript: str, video_id: str, 
                      channel_id: str, video_title: str) -> Dict:
        """Preview what would be ingested without actually ingesting.
        
        Useful for reviewing content before acceptance.
        """
        channel = self.curator.get_channel(channel_id)
        if not channel:
            return {"status": "error", "reason": "Channel not curated"}
        
        stats = self.processor.process_transcript_with_stats(
            transcript,
            video_id,
            channel.channel_name,
            video_title,
        )
        
        chunks = stats["chunks"]
        
        return {
            "status": "dry_run",
            "channel": channel.channel_name,
            "total_segments": stats["total_segments"],
            "total_chunks": len(chunks),
            "sample_chunks": [
                {
                    "topic": chunk.topic,
                    "type": chunk.content_type.value,
                    "content": chunk.content[:100] + "...",
                }
                for chunk in chunks[:3]  # Show first 3 as sample
            ],
        }
    
    def get_ingestion_stats(self) -> Dict:
        """Get cumulative ingestion statistics."""
        if not self.ingestion_results:
            return {"total_videos": 0, "total_chunks": 0}
        
        total_videos = len(self.ingestion_results)
        total_chunks = sum(r.get("chunks_ingested", 0) for r in self.ingestion_results)
        total_channels = len(set(r.get("channel") for r in self.ingestion_results))
        
        return {
            "total_videos_processed": total_videos,
            "total_chunks_ingested": total_chunks,
            "unique_channels": total_channels,
            "avg_chunks_per_video": total_chunks / total_videos if total_videos else 0,
        }


# Example: How to use
if __name__ == "__main__":
    sample_transcript = """
    In machine learning, supervised learning is a technique where the model learns 
    from labeled training data. The model receives input-output pairs and learns 
    the mapping between them. During prediction, the model applies this learned 
    mapping to new, unseen data.
    
    I think supervised learning is really cool. Anyway, let's look at common algorithms 
    like regression and classification. Regression predicts continuous values, meaning 
    the output is a real number. Classification assigns inputs to discrete categories 
    or classes.
    
    Thanks for watching, and don't forget to like and subscribe!
    """
    
    manager = YouTubeIngestionManager()
    
    # Dry run first
    print("[DRY RUN] Preview ingestion:")
    print("=" * 60)
    preview = manager.dry_run_review(
        sample_transcript,
        "abc123",
        "sentdex",
        "Machine Learning Basics"
    )
    print(f"Total chunks to ingest: {preview.get('total_chunks', 'N/A')}")
    print(f"Questions filtered: {preview.get('total_segments', 0) - preview.get('total_chunks', 0)}")
    
    if "sample_chunks" in preview:
        print("\nSample chunks:")
        for chunk in preview["sample_chunks"]:
            print(f"  - {chunk['topic']}: {chunk['content']}")
    
    # Now ingest
    print("\n[INGEST] Full ingestion:")
    print("=" * 60)
    report = manager.ingest_from_transcript(
        sample_transcript,
        "abc123",
        "sentdex",
        "Machine Learning Basics",
        "https://youtube.com/watch?v=abc123"
    )
    
    print(f"Status: {report['status']}")
    print(f"Channel: {report['channel']}")
    print(f"Chunks ingested: {report['chunks_ingested']} / {report['total_chunks']}")
    print(f"Ingestion rate: {report['ingestion_rate']}")
