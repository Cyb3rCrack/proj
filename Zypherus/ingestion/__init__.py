"""Ingestion modules (future split)."""

from .web import WebFetchResult, fetch_url_text, extract_text_from_html

__all__ = ["WebFetchResult", "fetch_url_text", "extract_text_from_html"]
