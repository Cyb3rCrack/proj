"""Curated YouTube channel ingestion for ACE.

Philosophy: Trusted educators only. Quality over quantity.
This prevents bloat and maintains knowledge integrity.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class CreatorTrust(Enum):
    """Trust level for content creators."""
    EXPERT = "expert"          # Domain expert, peer-reviewed, well-known
    EDUCATOR = "educator"      # Professional educator, clear teaching
    CONTRIBUTOR = "contributor"  # Good content but not verified expert
    UNVETTED = "unvetted"      # Not reviewed


@dataclass
class TrustedChannel:
    """A curated YouTube channel approved for ingestion."""
    channel_id: str
    channel_name: str
    owner: str                  # Creator name
    description: str
    expertise: List[str]        # Topics they cover: ["python", "algorithms", "ml"]
    trust_level: CreatorTrust
    max_videos_per_month: int   # Rate limit for ingestion
    video_quality_criteria: List[str]  # Must have captions, must be educational, etc.
    
    def should_ingest_video(self, video_title: str, has_captions: bool) -> bool:
        """Check if a video from this channel meets ingestion criteria."""
        if not has_captions and self.trust_level != CreatorTrust.EXPERT:
            return False  # Require captions for non-experts (lower confidence)
        
        # Exclude common low-value content
        excluded_keywords = ["reaction", "compilation", "ranking", "tier list", "review", "unboxing"]
        if any(kw in video_title.lower() for kw in excluded_keywords):
            return False
        
        return True


class YouTubeCurator:
    """Manages trusted channels and ingestion policy."""
    
    # Recommended starting set - all have excellent educational content
    RECOMMENDED_CHANNELS = {
        "sentdex": TrustedChannel(
            channel_id="sentdex",
            channel_name="Sentdex",
            owner="Harrison Kinsley",
            description="Python, machine learning, data science tutorials",
            expertise=["python", "machine-learning", "data-science", "neural-networks"],
            trust_level=CreatorTrust.EXPERT,
            max_videos_per_month=5,
            video_quality_criteria=["has_captions", "educational", "self_contained"],
        ),
        "3blue1brown": TrustedChannel(
            channel_id="3blue1brown",
            channel_name="3Blue1Brown",
            owner="Grant Sanderson",
            description="Math, algorithms, neural networks - exceptional visual teaching",
            expertise=["mathematics", "algorithms", "linear-algebra", "calculus", "neural-networks"],
            trust_level=CreatorTrust.EXPERT,
            max_videos_per_month=3,
            video_quality_criteria=["has_captions", "educational", "visually_clear"],
        ),
        "corey_schafer": TrustedChannel(
            channel_id="corey_schafer",
            channel_name="Corey Schafer",
            owner="Corey Schafer",
            description="Python programming from basics to advanced",
            expertise=["python", "django", "web-development", "programming-fundamentals"],
            trust_level=CreatorTrust.EDUCATOR,
            max_videos_per_month=4,
            video_quality_criteria=["has_captions", "educational"],
        ),
        "fireship": TrustedChannel(
            channel_id="fireship",
            channel_name="Fireship",
            owner="Jeff Delaney",
            description="Fast-paced coding tutorials and explanations",
            expertise=["web-dev", "javascript", "full-stack", "databases"],
            trust_level=CreatorTrust.EDUCATOR,
            max_videos_per_month=4,
            video_quality_criteria=["has_captions", "educational", "self_contained"],
        ),
        "mit_opencourseware": TrustedChannel(
            channel_id="mit_opencourseware",
            channel_name="MIT OpenCourseWare",
            owner="MIT",
            description="Official MIT course recordings - highest academic standard",
            expertise=["algorithms", "data-structures", "programming", "mathematics", "systems"],
            trust_level=CreatorTrust.EXPERT,
            max_videos_per_month=10,
            video_quality_criteria=["has_captions", "educational", "academic"],
        ),
        "computerphile": TrustedChannel(
            channel_id="computerphile",
            channel_name="Computerphile",
            owner="Dr. Mike Pound et al",
            description="Computer science concepts explained by academics",
            expertise=["algorithms", "security", "systems", "computational-theory"],
            trust_level=CreatorTrust.EXPERT,
            max_videos_per_month=5,
            video_quality_criteria=["has_captions", "educational"],
        ),
    }
    
    def __init__(self):
        self.channels: Dict[str, TrustedChannel] = {}
        self.ingestion_log: List[Dict] = []
    
    def add_trusted_channel(self, channel: TrustedChannel) -> None:
        """Add a channel to the trusted list."""
        self.channels[channel.channel_id] = channel
    
    def add_recommended_channels(self) -> None:
        """Load all recommended channels."""
        for channel_id, channel in self.RECOMMENDED_CHANNELS.items():
            self.add_trusted_channel(channel)
    
    def get_channel(self, channel_id: str) -> Optional[TrustedChannel]:
        """Retrieve channel configuration."""
        return self.channels.get(channel_id)
    
    def list_channels(self) -> List[TrustedChannel]:
        """List all curated channels."""
        return list(self.channels.values())
    
    def log_ingestion(self, channel_id: str, video_id: str, 
                      chunks_ingested: int, status: str = "success") -> None:
        """Log ingestible videos for rate limiting."""
        self.ingestion_log.append({
            "timestamp": datetime.now().isoformat(),
            "channel_id": channel_id,
            "video_id": video_id,
            "chunks_ingested": chunks_ingested,
            "status": status,
        })
    
    def get_channel_ingestion_count(self, channel_id: str, days: int = 30) -> int:
        """Get number of videos ingested from channel in last N days."""
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        count = 0
        for entry in self.ingestion_log:
            if entry["channel_id"] == channel_id:
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time > cutoff:
                    count += 1
        
        return count
    
    def can_ingest_from_channel(self, channel_id: str) -> bool:
        """Check if channel has available ingestion quota."""
        channel = self.get_channel(channel_id)
        if not channel:
            return False
        
        count = self.get_channel_ingestion_count(channel_id, days=30)
        return count < channel.max_videos_per_month


class YouTubeIngestionPolicy:
    """High-level policy for ACE's YouTube ingestion."""
    
    # Content requirements
    REQUIRE_CAPTIONS = True
    MIN_CONFIDENCE_FOR_INGESTION = 0.6  # 60% confidence threshold
    MIN_CHUNK_LENGTH = 100  # characters
    MAX_CHUNKS_PER_VIDEO = 20
    
    # Safety rules
    ALLOW_UNVETTED_CHANNELS = True  # Allow ingestion from non-curated channels
    REQUIRE_REVIEW_BEFORE_INGESTION = False  # Auto-ingest after filtering (optional)
    AUTO_REJECT_KEYWORDS = [
        "fake", "hoax", "scam", "clickbait",
        "unverified",
    ]
    
    @staticmethod
    def check_policy_compliance(
        channel_trust: CreatorTrust,
        has_captions: bool,
        confidence: float,
        auto_reject_found: bool,
    ) -> bool:
        """Check if content meets ingestion policy."""
        # Never ingest from unvetted unless explicitly allowed
        if channel_trust == CreatorTrust.UNVETTED and not YouTubeIngestionPolicy.ALLOW_UNVETTED_CHANNELS:
            return False
        
        # Require captions for low-trust creators
        if YouTubeIngestionPolicy.REQUIRE_CAPTIONS and not has_captions:
            if channel_trust not in [CreatorTrust.EXPERT, CreatorTrust.EDUCATOR]:
                return False
        
        # Confidence threshold
        if confidence < YouTubeIngestionPolicy.MIN_CONFIDENCE_FOR_INGESTION:
            return False
        
        # Safety check
        if auto_reject_found:
            return False
        
        return True


if __name__ == "__main__":
    curator = YouTubeCurator()
    curator.add_recommended_channels()
    
    print("Recommended Trusted Channels for ACE:")
    print("=" * 60)
    
    for channel in curator.list_channels():
        print(f"\n{channel.channel_name} (@{channel.owner})")
        print(f"  Trust Level: {channel.trust_level.value}")
        print(f"  Expertise: {', '.join(channel.expertise)}")
        print(f"  Max videos/month: {channel.max_videos_per_month}")
        print(f"  Criteria: {', '.join(channel.video_quality_criteria)}")
