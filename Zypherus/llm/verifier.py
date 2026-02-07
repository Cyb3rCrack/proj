"""Answer verification against source documents."""

import re
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def verify_answer(answer: str, 
                 context_chunks: List[Dict[str, Any]],
                 llm_generate_fn=None,
                 max_chunk_chars: int = 1000,
                 max_tokens: int = 256) -> Dict[str, Any]:
    """Verify answer claims against context documents.
    
    Args:
        answer: Answer text to verify
        context_chunks: List of context documents with 'source' and 'text'
        llm_generate_fn: Function to generate LLM verification (optional)
        max_chunk_chars: Max characters per chunk in context
        max_tokens: Max tokens for LLM verification response
    
    Returns:
        Verification result with confidence and unsupported claims
    """
    if not llm_generate_fn:
        # Fallback to basic keyword matching if no LLM available
        return _fallback_verify(answer, context_chunks)
    
    # Prepare context
    context = "\n\n".join(
        f"[Source: {c.get('source', 'unknown')}] {c.get('text', '')[:max_chunk_chars]}"
        for c in context_chunks
    )
    
    # Create verification prompt
    prompt = f"""Verify if the following answer is supported by the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with a JSON object containing:
1. confidence: number between 0.0 and 1.0
2. unsupported_claims: list of claims not in context
3. reasoning: brief explanation

Return only valid JSON."""
    
    try:
        response = llm_generate_fn(prompt, max_tokens=max_tokens)
        if not response:
            logger.warning("Verification unavailable (LLM offline)")
            return {
                "confidence": None,
                "unsupported_claims": [],
                "status": "unavailable"
            }
        
        # Try to parse JSON
        result = _parse_verification_response(response)
        if result:
            logger.debug(f"Verification: confidence={result.get('confidence')}")
            return result
        
        # Fallback to regex parsing
        return _fallback_parse_verification(response)
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {
            "confidence": None,
            "unsupported_claims": [],
            "status": "error",
            "error": str(e)
        }


def _fallback_verify(answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic keyword-based verification without LLM."""
    context_text = " ".join(
        c.get("text", "") for c in context_chunks
    ).lower()
    
    answer_lower = answer.lower()
    # Simple heuristic: if answer keywords appear in context
    tokens = re.findall(r"\b\w+\b", answer_lower)
    found_tokens = sum(1 for t in tokens if t in context_text) if tokens else 0
    confidence = found_tokens / len(tokens) if tokens else 0.5
    
    return {
        "confidence": min(1.0, max(0.0, confidence)),
        "unsupported_claims": [],
        "status": "fallback",
        "method": "keyword_matching"
    }


def _parse_verification_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from verification response."""
    try:
        # Try direct JSON parsing
        obj = json.loads(response)
        if isinstance(obj, dict):
            return _validate_verification_json(obj)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            return _validate_verification_json(obj)
        except json.JSONDecodeError:
            pass
    
    return None


def _validate_verification_json(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize verification JSON."""
    if not isinstance(obj, dict):
        return None
    
    try:
        confidence = obj.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            confidence = min(1.0, max(0.0, confidence))
        
        return {
            "confidence": confidence,
            "unsupported_claims": obj.get("unsupported_claims", []),
            "reasoning": obj.get("reasoning", ""),
            "status": "verified"
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid verification JSON: {e}")
        return None


def _fallback_parse_verification(response: str) -> Dict[str, Any]:
    """Extract confidence and unsupported claims from free text."""
    # Try to find a confidence value
    confidence = 0.5
    conf_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|/100)?", response)
    if conf_match:
        try:
            val = float(conf_match.group(1))
            if 0 <= val <= 100:
                confidence = val / 100.0
            elif 0 <= val <= 1:
                confidence = val
        except ValueError:
            pass
    
    # Check for unsupported claims
    unsupported = []
    if any(phrase in response.lower() for phrase in 
           ["unsupported", "not supported", "not found", "contradicts"]):
        unsupported.append("model indicates unsupported or contradictory claims")
    
    return {
        "confidence": confidence,
        "unsupported_claims": unsupported,
        "status": "fallback_parsed",
        "method": "regex_extraction"
    }

