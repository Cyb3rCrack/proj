"""Lightweight web search helpers."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .web import DEFAULT_USER_AGENT


@dataclass
class SearchResult:
    url: str
    title: str


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    url = url.split("#", 1)[0]
    url = url.rstrip("/")
    return url


def _extract_ddg_results(html_text: str) -> List[SearchResult]:
    results: List[SearchResult] = []
    # DuckDuckGo HTML results use anchors with class "result__a".
    pattern = re.compile(r"<a[^>]+class=\"result__a\"[^>]+href=\"(.*?)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    for href, title_html in pattern.findall(html_text or ""):
        title = re.sub(r"<[^>]+>", "", title_html or "")
        title = html.unescape(title).strip()
        url = html.unescape(href or "").strip()

        # DuckDuckGo wraps outbound links with a redirect.
        if "duckduckgo.com/l/" in url:
            try:
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                if "uddg" in params and params["uddg"]:
                    url = unquote(params["uddg"][0])
            except Exception:
                pass

        if not url:
            continue

        results.append(SearchResult(url=url, title=title))

    return results


def _filter_results_by_domain(results: Sequence[SearchResult], allowed_domains: Optional[Iterable[str]]) -> List[SearchResult]:
    if not allowed_domains:
        return list(results)

    allowed = {d.lower() for d in allowed_domains if d}
    filtered: List[SearchResult] = []
    for r in results:
        try:
            host = (urlparse(r.url).netloc or "").lower()
        except Exception:
            continue
        if not host:
            continue
        if any(host == d or host.endswith("." + d) for d in allowed):
            filtered.append(r)
    return filtered


def search_web(
    query: str,
    *,
    max_results: int = 8,
    timeout_s: float = 10.0,
    user_agent: Optional[str] = None,
    allowed_domains: Optional[Iterable[str]] = None,
) -> List[SearchResult]:
    """Search the web using DuckDuckGo HTML results.

    Returns a list of SearchResult objects with URLs and titles.
    """
    query = (query or "").strip()
    if not query:
        return []

    headers = {
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    html_text = response.text or ""

    results = _extract_ddg_results(html_text)
    results = _filter_results_by_domain(results, allowed_domains)

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[SearchResult] = []
    for r in results:
        normalized = _normalize_url(r.url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(SearchResult(url=normalized, title=r.title))
        if len(deduped) >= max_results:
            break

    return deduped


__all__ = ["SearchResult", "search_web"]
