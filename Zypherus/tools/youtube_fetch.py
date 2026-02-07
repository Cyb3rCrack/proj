"""Fetch YouTube transcripts and metadata directly from URLs.

Uses yt-dlp to extract subtitles and metadata without API keys.
"""

import re
import json
import os
from typing import Optional, Dict, Tuple
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL.
    
    Handles:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/watch?v=VIDEO_ID&list=...
    """
    # Standard URL format
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None


def map_channel_to_curator_id(channel_name: str) -> Optional[str]:
    """Map YouTube channel names to curator channel IDs.
    
    Maps human-readable channel names (from yt-dlp metadata) to the 
    internal curator channel IDs.
    """
    # Mapping from channel name (from YouTube metadata) to curator ID
    channel_mapping = {
        'Corey Schafer': 'corey_schafer',
        'Sentdex': 'sentdex',
        '3Blue1Brown': '3blue1brown',
        'Fireship': 'fireship',
        'MIT OpenCourseWare': 'mit_opencourseware',
        'Computerphile': 'computerphile',
        # Add fuzzy matches for common variations
        'corey schafer': 'corey_schafer',
        'sentdex': 'sentdex',
        '3blue1brown': '3blue1brown',
        'fireship': 'fireship',
        'mit': 'mit_opencourseware',
        'computerphile': 'computerphile',
    }
    
    # Try exact match first
    if channel_name in channel_mapping:
        return channel_mapping[channel_name]
    
    # Try case-insensitive match
    channel_lower = channel_name.lower()
    for key, value in channel_mapping.items():
        if key.lower() == channel_lower:
            return value
    
    # Try substring match for flexibility
    channel_lower = channel_name.lower()
    for key, value in channel_mapping.items():
        if key.lower() in channel_lower or channel_lower in key.lower():
            return value
    
    return None


def fetch_transcript(video_id: str, output_file: Optional[str] = None) -> Optional[Tuple[str, Dict]]:
    """Fetch transcript and metadata from YouTube using yt-dlp.
    
    Args:
        video_id: YouTube video ID
        output_file: Optional file path to save transcript
        
    Returns:
        (transcript_text, metadata) or None if fetch fails
        
    Metadata includes:
        - title: Video title
        - channel: Channel name
        - duration: Video duration in seconds
        - upload_date: Publication date
    """
    if not yt_dlp:
        print("[YOUTUBE] Error: yt-dlp not installed")
        return None
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        print("[YOUTUBE] Fetching video information and transcript...")
        
        # Configure yt-dlp options
        ydl_opts = {
            'quiet': False,
            'no_warnings': False,
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'socket_timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            info = ydl.extract_info(url, download=False)
        
        # Extract metadata
        metadata = {
            'title': info.get('title', 'Unknown'),
            'channel': info.get('uploader', 'Unknown'),
            'channel_id': info.get('channel_id', 'unknown'),
            'duration': info.get('duration', 0),
            'upload_date': info.get('upload_date', ''),
        }
        
        # Extract transcript from subtitles
        transcript = None
        
        # Check for auto-generated captions first
        if 'automatic_captions' in info and info['automatic_captions']:
            for lang_code in ['en', 'en-US', 'en-GB']:
                if lang_code in info['automatic_captions']:
                    captions = info['automatic_captions'][lang_code]
                    for caption in captions:  # type: ignore
                        if caption.get('ext') == 'vtt':
                            try:
                                import urllib.request
                                with urllib.request.urlopen(caption['url'], timeout=10) as response:
                                    vtt_content = response.read().decode('utf-8')
                                    transcript = clean_vtt(vtt_content)
                                    break
                            except Exception as e:
                                print(f"[YOUTUBE] Warning: Could not fetch auto-captions: {e}")
                    if transcript:
                        break
        
        # Fall back to regular subtitles
        if not transcript and 'subtitles' in info and info['subtitles']:
            for lang_code in ['en', 'en-US', 'en-GB']:
                if lang_code in info['subtitles']:
                    captions = info['subtitles'][lang_code]
                    for caption in captions:  # type: ignore
                        if caption.get('ext') == 'vtt':
                            try:
                                import urllib.request
                                with urllib.request.urlopen(caption['url'], timeout=10) as response:
                                    vtt_content = response.read().decode('utf-8')
                                    transcript = clean_vtt(vtt_content)
                                    break
                            except Exception as e:
                                print(f"[YOUTUBE] Warning: Could not fetch captions: {e}")
                    if transcript:
                        break
        
        if transcript:
            # Save to file if requested
            if output_file:
                Path(output_file).write_text(transcript)
            
            print(f"[YOUTUBE] Successfully fetched {len(transcript)} characters from '{metadata['title']}'")
            return transcript, metadata
        else:
            print(f"[YOUTUBE] Warning: No captions available for this video")
            return None
            
    except Exception as e:
        print(f"[YOUTUBE] Error fetching transcript: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_vtt(vtt_content: str) -> str:
    """Convert VTT subtitle format to plain text.
    
    Removes:
    - WEBVTT header
    - Timestamp lines (00:00:00.000 --> 00:00:05.000)
    - Formatting tags and timing codes
    - Duplicate consecutive lines
    """
    import re
    
    # Remove XML-style timing tags like <00:00:00.359>
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', vtt_content)
    
    # Remove caption formatting tags like <c>...
    text = re.sub(r'<[^>]+>', '', text)
    
    lines = text.split('\n')
    text_lines = []
    prev_line = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip header
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        
        # Skip timestamp lines (standard VTT format)
        if '-->' in line:
            continue
        
        # Skip lines that are just timestamps
        if re.match(r'^\d{2}:\d{2}:\d{2}', line):
            continue
        
        # Add non-empty lines, but skip duplicates
        if line and line != prev_line:
            text_lines.append(line)
            prev_line = line
    
    # Join with spaces
    text = ' '.join(text_lines)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences for readability
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    return text.strip()


def ingest_from_youtube_url(ace, url: str, channel_id: Optional[str] = None) -> Dict:
    """Complete workflow: URL → transcript → ingestion.
    
    Args:
        ace: Zypherus instance
        url: YouTube URL
        channel_id: Optional channel ID (auto-detected if not provided)
        
    Returns:
        Ingestion report dict
    """
    from .youtube_ace import YouTubeIngestionManager
    
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        return {
            'status': 'error',
            'reason': 'Invalid YouTube URL',
            'url': url
        }
    
    # Fetch transcript and metadata
    result = fetch_transcript(video_id)
    if not result:
        return {
            'status': 'error',
            'reason': 'Could not fetch transcript',
            'video_id': video_id
        }
    
    transcript, metadata = result
    
    # Map channel name to curator ID, or use provided channel_id
    if not channel_id:
        channel_name = metadata.get('channel', '')
        mapped_id = map_channel_to_curator_id(channel_name)
        if mapped_id:
            channel_id = mapped_id
            print(f"[YOUTUBE] Mapped '{channel_name}' -> '{channel_id}'")
        else:
            # Fall back to YouTube channel ID (will likely be rejected but try anyway)
            channel_id = metadata.get('channel_id', 'unknown')
            print(f"[YOUTUBE] Warning: Channel '{channel_name}' not in curator mapping, using YouTube ID: {channel_id}")
    
    # Ingest using manager
    manager = YouTubeIngestionManager(ace)
    report = manager.ingest_from_transcript(
        transcript=transcript,
        video_id=video_id,
        channel_id=str(channel_id or 'unknown'),
        video_title=metadata.get('title', 'Unknown'),
        video_url=url,
        channel_name=metadata.get('channel', None)
    )
    
    return report
