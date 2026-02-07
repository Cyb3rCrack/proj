#!/usr/bin/env python3
"""
Zypherus API Client - Example usage for deployed instance
"""

import requests
import json
from typing import Optional, Dict, Any


class ZypherusClient:
    """Simple client for interacting with deployed Zypherus API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of deployed Zypherus (e.g., https://zypherus.onrender.com)
            api_key: Optional API key if authentication is enabled
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Ask a question."""
        data = {"question": question}
        resp = self.session.post(
            f"{self.base_url}/api/answer",
            json=data,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    
    def ingest(self, text: str, source: str = "api") -> Dict[str, Any]:
        """Ingest new knowledge."""
        data = {"text": text, "source": source}
        resp = self.session.post(
            f"{self.base_url}/api/ingest",
            json=data,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        resp = self.session.get(f"{self.base_url}/api/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def get_beliefs(self) -> Dict[str, Any]:
        """Get top beliefs."""
        resp = self.session.get(f"{self.base_url}/api/beliefs", timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search memory entries."""
        data = {"query": query, "limit": min(limit, 100)}
        resp = self.session.post(
            f"{self.base_url}/api/search",
            json=data,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_memory(self) -> Dict[str, Any]:
        """Get memory info."""
        resp = self.session.get(f"{self.base_url}/api/memory", timeout=10)
        resp.raise_for_status()
        return resp.json()


def demo():
    """Demo usage."""
    # Replace with your deployed URL
    client = ZypherusClient("https://zypherus.onrender.com")
    
    print("Zypherus API Client Demo\n")
    
    # Check health
    print("Checking server health...")
    if not client.health_check():
        print("Server is not responding!")
        return
    print("Server is healthy\n")
    
    # Get status
    print("System Status:")
    status = client.get_status()
    print(f"  Memory entries: {status['data'].get('memory_entries', 0)}")
    print(f"  Claims: {status['data'].get('claims', 0)}")
    print(f"  Concepts: {status['data'].get('concepts', 0)}\n")
    
    # Ask a question
    print("Asking a question...")
    try:
        answer = client.answer("What is machine learning?")
        print(f"Answer: {answer['data']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Get beliefs
    print("Top Beliefs:")
    try:
        beliefs = client.get_beliefs()
        for claim in beliefs['data'].get('top_claims', [])[:3]:
            print(f"  • {claim['subject']} {claim['predicate']} {claim['object']}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Search
    print("Searching for 'engineering'...")
    try:
        results = client.search("engineering", limit=5)
        total = results['data'].get('total_entries', 0)
        print(f"  Found {total} entries\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("Demo complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        client = ZypherusClient(url)
        
        if client.health_check():
            print(f"Connected to {url}")
            status = client.get_status()
            print(f"Status: {json.dumps(status, indent=2)}")
        else:
            print(f"✗ Cannot reach {url}")
    else:
        demo()
