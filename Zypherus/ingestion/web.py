"""Web ingestion helpers."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "nav",
    "header",
    "footer",
    "form",
    "input",
    "button",
    "aside",
    "template",
    "meta",
}

# Tags that prioritize content extraction (higher priority in main flow)
PRIORITY_CONTENT_TAGS = {
    "article",
    "main",
    "section",
}

BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "br",
    "hr",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


@dataclass
class WebFetchResult:
    text: str
    url: str
    title: Optional[str]
    content_type: str
    status_code: int
    source_hint: str


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._title_chunks: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._in_title:
            self._title_chunks.append(data)
            return
        if self._skip_depth > 0:
            return
        self._chunks.append(data)

    def get_text(self) -> Tuple[str, Optional[str]]:
        raw_text = "".join(self._chunks)
        title = "".join(self._title_chunks).strip() or None
        return raw_text, title


def _normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _default_source_from_url(url: str, title: Optional[str]) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "unknown"
    path = (parsed.path or "").rstrip("/")
    base = f"web:{host}{path}"
    if title:
        short_title = re.sub(r"[^A-Za-z0-9 _-]+", "", title).strip()
        if short_title:
            short_title = re.sub(r"\s+", "-", short_title)[:60]
            base = f"{base}#{short_title}"
    return base


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")
    if not parsed.netloc:
        raise ValueError("URL must include a hostname")


def _extract_text_with_regex(html_text: str) -> Tuple[str, Optional[str]]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text or "", flags=re.IGNORECASE | re.DOTALL)
    title = _normalize_text(title_match.group(1)) if title_match else None

    cleaned = re.sub(r"<!--.*?-->", " ", html_text or "", flags=re.DOTALL)
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<noscript[^>]*>.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)

    block_pattern = r"</?(?:p|div|section|article|main|li|ul|ol|br|hr|table|tr|td|th|h[1-6]|header|footer|nav|aside)[^>]*>"
    cleaned = re.sub(block_pattern, "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    return _normalize_text(cleaned), title


def extract_text_from_html(html_text: str) -> Tuple[str, Optional[str]]:
    parser = _HTMLTextExtractor()
    parser.feed(html_text or "")
    raw_text, title = parser.get_text()
    normalized = _normalize_text(raw_text)

    if len(normalized) < 200:
        fallback_text, fallback_title = _extract_text_with_regex(html_text)
        if fallback_title and not title:
            title = fallback_title
        normalized = fallback_text

    return normalized, title


def fetch_url_text(
    url: str,
    *,
    timeout_s: float = 20.0,
    max_chars: int = 200000,
    user_agent: Optional[str] = None,
) -> WebFetchResult:
    """Fetch URL and extract text intelligently.
    
    Args:
        max_chars: Maximum characters to extract (200k = ~40-50k words = full articles)
    """
    _validate_url(url)
    headers = {
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    response = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)

    if response.status_code == 403:
        parsed = urlparse(url)
        if parsed.netloc.endswith("medium.com"):
            stripped_url = url.replace("https://", "").replace("http://", "")
            proxy_url = f"https://r.jina.ai/http://{stripped_url}"
            response = requests.get(proxy_url, headers=headers, timeout=timeout_s, allow_redirects=True)
    response.raise_for_status()

    if response.encoding is None:
        response.encoding = response.apparent_encoding or "utf-8"

    content_type = (response.headers.get("content-type") or "").lower()
    if content_type and "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}")
    body_text = response.text or ""

    if "html" in content_type:
        extracted, title = extract_text_from_html(body_text)
    else:
        extracted = _normalize_text(body_text)
        title = None

    if max_chars and len(extracted) > max_chars:
        # Soft limit: try to preserve paragraph boundaries
        extracted = extracted[:max_chars]
        # Trim to last sentence boundary
        last_period = extracted.rfind(".")
        if last_period > max_chars * 0.95:  # Only if close to limit
            extracted = extracted[:last_period + 1]
        extracted = extracted.rstrip()

    source_hint = _default_source_from_url(url, title)

    return WebFetchResult(
        text=extracted,
        url=url,
        title=title,
        content_type=content_type,
        status_code=response.status_code,
        source_hint=source_hint,
    )


__all__ = ["WebFetchResult", "fetch_url_text", "extract_text_from_html"]
