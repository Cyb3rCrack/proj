"""Autonomous web crawler for discovering and ingesting coding knowledge.

Optimized for speed without compromising safety:
- Parallel requests (ThreadPoolExecutor, 3-6x speedup)
- lxml parser (2-3x faster than html.parser)
- Early URL deduplication (reduces wasted HTTP calls)
- Smart text extraction (main/article tags only)
- Robots.txt enforcement
"""

import re
import time
import logging
import os
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.robotparser as robotparser
except ImportError:
    requests = None
    BeautifulSoup = None
    robotparser = None
    BeautifulSoup = None
    robotparser = None

logger = logging.getLogger("ZYPHERUS.Crawler")


@dataclass
class CodingSource:
    """A trusted source of coding knowledge."""
    name: str
    domain: str
    base_url: str
    topics: List[str]  # e.g., ["python", "algorithms", "ml"]
    starting_urls: List[str]  # Entry points
    exclude_patterns: List[str] = field(default_factory=list)  # Regex patterns to skip


# Curated high-quality coding sources
TRUSTED_CODING_SOURCES = [
    CodingSource(
        name="Python Official Docs",
        domain="python.org",
        base_url="https://docs.python.org/3/",
        topics=["python", "language-basics", "standard-library"],
        starting_urls=[
            "https://docs.python.org/3/library/",
            "https://docs.python.org/3/tutorial/",
        ],
        exclude_patterns=[r"\.po$", r"search\?", r"genindex"],
    ),
    CodingSource(
        name="MDN Web Docs",
        domain="mdn.org",
        base_url="https://developer.mozilla.org/",
        topics=["javascript", "web-dev", "html-css"],
        starting_urls=[
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/",
            "https://developer.mozilla.org/en-US/docs/Learn/",
        ],
        exclude_patterns=[r"search\?", r"interactive-example"],
    ),
    CodingSource(
        name="Real Python",
        domain="realpython.com",
        base_url="https://realpython.com/",
        topics=["python", "tutorials", "best-practices"],
        starting_urls=[
            "https://realpython.com/tutorials/",
            "https://realpython.com/learning-paths/",
        ],
        exclude_patterns=[r"^https://realpython\.com/courses", r"\?.*"],
    ),
    CodingSource(
        name="GeeksforGeeks",
        domain="geeksforgeeks.org",
        base_url="https://www.geeksforgeeks.org/",
        topics=["algorithms", "data-structures", "programming"],
        starting_urls=[
            "https://www.geeksforgeeks.org/fundamentals/",
            "https://www.geeksforgeeks.org/dsa/",
        ],
        exclude_patterns=[r"/forum/", r"/share\.php"],
    ),
    CodingSource(
        name="Stack Overflow",
        domain="stackoverflow.com",
        base_url="https://stackoverflow.com/",
        topics=["q-and-a", "debugging", "best-practices"],
        starting_urls=[
            "https://stackoverflow.com/questions/tagged/python?tab=newest",
            "https://stackoverflow.com/questions/tagged/javascript?tab=newest",
        ],
        exclude_patterns=[r"/users/", r"/review/"],
    ),
    CodingSource(
        name="GitHub Guides",
        domain="github.com/guides",
        base_url="https://guides.github.com/",
        topics=["git", "github", "collaboration"],
        starting_urls=[
            "https://guides.github.com/",
        ],
        exclude_patterns=[r"search", r"trending"],
    ),
]


class WebCrawler:
    """Optimized parallel crawler for discovering coding knowledge."""
    
    def __init__(self, ace_instance=None, rate_limit_s: float = 0.2, workers: int = 4):
        """
        Args:
            ace_instance: ACE instance to use for ingestion
            rate_limit_s: Seconds to wait between requests (reduced for parallel)
            workers: Number of parallel threads (4-8 recommended)
        """
        self.ace = ace_instance
        self.rate_limit_s = rate_limit_s
        self.workers = workers

        fast_mode = os.getenv("ZYPHERUS_INGEST_FAST", "true").lower() in ("1", "true", "yes")
        
        workers_env = os.getenv("ZYPHERUS_CRAWL_WORKERS")
        if workers_env is not None:
            try:
                self.workers = int(workers_env)
            except ValueError:
                logger.warning(f"Invalid ZYPHERUS_CRAWL_WORKERS={workers_env}")
        elif fast_mode:
            self.workers = max(self.workers, 8)

        rate_limit_env = os.getenv("ZYPHERUS_CRAWL_RATE_LIMIT_S")
        if rate_limit_env is not None:
            try:
                self.rate_limit_s = float(rate_limit_env)
            except ValueError:
                logger.warning(f"Invalid ZYPHERUS_CRAWL_RATE_LIMIT_S={rate_limit_env}")
        elif fast_mode:
            self.rate_limit_s = min(self.rate_limit_s, 0.05)
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.robot_parsers = {}  # Cache robots.txt per domain
        self.max_tracked_urls = 10000  # MEMORY LEAK FIX: Prevent unbounded growth
        
        if not requests or not BeautifulSoup:
            logger.warning("requests and BeautifulSoup not installed")
    
    def _get_robots_parser(self, domain: str) -> Optional[object]:
        """Get cached robots.txt parser for domain."""
        if not robotparser:
            return None
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        
        try:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            rp.read()
            self.robot_parsers[domain] = rp
            return rp
        except Exception:
            return None
    
    def _can_fetch(self, url: str, domain: str) -> bool:
        """Check if URL can be fetched per robots.txt."""
        try:
            rp = self._get_robots_parser(domain)
            if rp and hasattr(rp, 'can_fetch'):
                return rp.can_fetch("*", url)  # type: ignore
        except Exception:
            pass
        return True
    
    def clear_url_history(self):
        """MEMORY LEAK FIX: Clear URL tracking to free memory between crawls."""
        self.visited_urls.clear()
        self.failed_urls.clear()
    
    def _trim_url_history(self):
        """MEMORY LEAK FIX: Keep URL tracking within memory bounds."""
        total = len(self.visited_urls) + len(self.failed_urls)
        if total > self.max_tracked_urls:
            if len(self.failed_urls) > self.max_tracked_urls // 2:
                self.failed_urls.clear()
            if len(self.visited_urls) > self.max_tracked_urls * 0.9:
                self.visited_urls = set(list(self.visited_urls)[len(self.visited_urls)//2:])
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to reduce duplicates."""
        url = url.split("#")[0]  # Remove fragments
        url = url.rstrip("/")     # Remove trailing slash
        return url
    
    def _extract_text_smart(self, soup) -> Tuple[str, bool]:  # soup: BeautifulSoup if available
        """Extract text from main content areas (faster, better signal).
        
        Returns:
            (text, is_substantial)
        """
        # Try main content areas first
        main = soup.find("main") or soup.find("article") or soup.find("section", class_=re.compile("content|main"))
        
        if main:
            # Remove nav/footer from main content
            for tag in main(['script', 'style', 'nav', 'footer', 'aside']):
                tag.decompose()
            text = main.get_text(separator=' ', strip=True)
        else:
            # Fallback: extract from entire page
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
        
        # Early cutoff: only process substantial pages
        word_count = len(text.split())
        is_substantial = word_count > 100
        
        # Truncate to save CPU (can be reprocessed during belief building)
        if len(text) > 20000:
            text = text[:20000]
        
        return text, is_substantial
    
    def _fetch_and_process(self, url: str, source: CodingSource) -> Optional[Tuple[str, str]]:
        """Fetch, parse, and extract text from URL.
        
        Returns:
            (source_name, text) or None if failed
        """
        url = self._normalize_url(url)
        
        # Early dedup before HTTP call
        if url in self.visited_urls or url in self.failed_urls:
            return None
        
        # MEMORY LEAK FIX: Prevent unbounded URL tracking
        self._trim_url_history()
        
        # Check robots.txt
        domain = urlparse(url).netloc
        if not self._can_fetch(url, domain):
            logger.debug(f"Robots.txt blocks: {url}")
            self.failed_urls.add(url)
            return None
        
        try:
            if requests is None:
                logger.warning(f"requests library not available, skipping {url}")
                self.failed_urls.add(url)
                return None
                
            response = requests.get(url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            # Use lxml for 2-3x faster parsing
            if BeautifulSoup is None:
                logger.warning(f"BeautifulSoup not available, skipping {url}")
                self.failed_urls.add(url)
                return None
                
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Smart extraction
            text, is_substantial = self._extract_text_smart(soup)
            
            if not is_substantial:
                self.visited_urls.add(url)
                return None
            
            source_name = f"{source.name}:{url}"
            self.visited_urls.add(url)
            return (source_name, text)
            
        except Exception as e:
            logger.debug(f"Fetch failed {url}: {type(e).__name__}")
            self.failed_urls.add(url)
            return None
    
    def crawl_source(self, source: CodingSource, max_pages: int = 10) -> int:
        """Crawl source using parallel requests.
        
        Args:
            source: CodingSource to crawl
            max_pages: Maximum pages to ingest from this source
            
        Returns:
            Number of pages ingested
        """
        if not self.ace:
            logger.error("No ACE instance provided")
            return 0
        
        ingested = 0
        to_visit = list(source.starting_urls)
        visited_from_source = set()
        
        logger.info(f"Starting crawl of {source.name} (workers={self.workers})")
        
        # Parallel fetching
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while to_visit and len(visited_from_source) < max_pages:
                # Submit batch of URLs
                batch_urls = []
                while to_visit and len(batch_urls) < self.workers * 2 and len(visited_from_source) < max_pages:
                    url = to_visit.pop(0)
                    url = self._normalize_url(url)
                    
                    # Skip already visited
                    if url in visited_from_source:
                        continue
                    
                    # Check exclude patterns
                    if source.exclude_patterns:
                        if any(re.search(p, url) for p in source.exclude_patterns):
                            continue
                    
                    batch_urls.append(url)
                
                if not batch_urls:
                    break
                
                # Submit all URLs in parallel
                futures = {
                    executor.submit(self._fetch_and_process, url, source): url
                    for url in batch_urls
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            source_name, text = result
                            
                            # Ingest
                            try:
                                self.ace.ingest_document(
                                    source_name,
                                    text,
                                    skip_filter=True,
                                    chunk_size=900,  # Larger chunks during crawl
                                )
                                ingested += 1
                                logger.info(f"Ingested: {len(text)} chars")
                            except Exception as e:
                                logger.warning(f"Ingest failed: {e}")
                            
                            visited_from_source.add(futures[future])
                    except Exception as e:
                        logger.debug(f"Future failed: {e}")
                
                # Gentle rate limit between batches
                time.sleep(self.rate_limit_s)
        
        logger.info(f"Completed {source.name}: {ingested} pages")
        return ingested
    
    def crawl_all_sources(self, max_pages_per_source: int = 10) -> int:
        """Crawl all trusted coding sources.
        
        Args:
            max_pages_per_source: Max pages to ingest per source
            
        Returns:
            Total pages ingested
        """
        total = 0
        for source in TRUSTED_CODING_SOURCES:
            try:
                count = self.crawl_source(source, max_pages=max_pages_per_source)
                total += count
            except Exception as e:
                logger.error(f"Error crawling {source.name}: {e}")
                continue
        
        logger.info(f"\n=== Autonomous Crawl Complete ===")
        logger.info(f"Total pages ingested: {total}")
        logger.info(f"Total URLs visited: {len(self.visited_urls)}")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")
        
        return total
    
    def crawl_by_topic(self, topics: List[str], max_pages_per_source: int = 5) -> int:
        """Crawl only sources matching specific topics.
        
        Args:
            topics: List of topics to crawl (e.g., ["python", "ml"])
            max_pages_per_source: Max pages per matching source
            
        Returns:
            Total pages ingested
        """
        matching_sources = [
            src for src in TRUSTED_CODING_SOURCES
            if any(topic in src.topics for topic in topics)
        ]
        
        logger.info(f"Crawling sources for topics: {topics}")
        logger.info(f"Found {len(matching_sources)} matching sources")
        
        total = 0
        for source in matching_sources:
            try:
                count = self.crawl_source(source, max_pages=max_pages_per_source)
                total += count
            except Exception as e:
                logger.error(f"Error crawling {source.name}: {e}")
                continue
        
        return total


__all__ = ["WebCrawler", "TRUSTED_CODING_SOURCES", "CodingSource"]
