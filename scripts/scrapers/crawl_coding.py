"""Standalone script to run autonomous web crawler.

Usage:
    python crawl_coding.py all              # Crawl all sources
    python crawl_coding.py python           # Crawl only Python sources
    python crawl_coding.py ml algorithms    # Crawl ML and algorithms sources
"""

import sys
from Zypherus.core.ace import ACE
from Zypherus.tools.web_crawler import WebCrawler


def main():
    # Initialize ACE
    print("[ACE] Initializing...")
    ace = ACE()
    
    # Create crawler (with parallelization: 4 workers, 0.2s rate limit between batches)
    crawler = WebCrawler(ace_instance=ace, rate_limit_s=0.2)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        topics = sys.argv[1:]
        
        if topics[0] == "all":
            print("\n[CRAWLER] Starting parallel autonomous crawl of ALL sources...")
            print("[CRAWLER] Using ThreadPoolExecutor with 4 workers + lxml parser")
            print("[CRAWLER] This will crawl: Python Docs, MDN, Real Python, GeeksforGeeks, Stack Overflow, GitHub")
            max_pages = 15  # Increased - now much faster with parallelization
            total = crawler.crawl_all_sources(max_pages_per_source=max_pages)
        else:
            print(f"\n[CRAWLER] Starting parallel crawl for topics: {topics}")
            print("[CRAWLER] Using ThreadPoolExecutor with 4 workers + lxml parser")
            max_pages = 20  # Increased - comprehensive crawl now feasible
            total = crawler.crawl_by_topic(topics, max_pages_per_source=max_pages)
    else:
        print("[CRAWLER] No topics specified. Usage: python crawl_coding.py [all|python|javascript|algorithms|ml]")
        print("[CRAWLER] Available topics: python, javascript, web-dev, algorithms, data-structures, ml, tutorials, best-practices")
        return
    
    print(f"\nSuccessfully ingested {total} pages into Zypherus's memory")
    if total > 0:
        print("[CRAWLER] You can now ask Zypherus questions about coding and it will use this knowledge!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CRAWLER] Crawl interrupted by user")
    except Exception as e:
        print(f"\n[CRAWLER] Error: {e}")
        print("\nMake sure beautifulsoup4 is installed:")
        print("  pip install beautifulsoup4")
