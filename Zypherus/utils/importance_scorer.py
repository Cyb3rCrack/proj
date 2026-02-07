"""Score importance/knowledge value of text chunks.

Determines which content is worth storing vs just processing.
Higher score = more foundational/important knowledge.
"""

import re
from typing import Optional

class ImportanceScorer:
    """Score chunks by knowledge importance."""
    
    # Knowledge indicators (boost importance)
    TECHNICAL_TERMS = {
        "algorithm", "data structure", "function", "class", "method",
        "implementation", "design pattern", "architecture", "framework",
        "protocol", "specification", "standard", "definition", "concept",
        "theory", "principle", "law", "rule", "axiom", "theorem",
    }
    
    DEFINITIVE_VERBS = {
        "is", "are", "defines", "means", "refers to", "denotes",
        "consists of", "comprises", "includes", "contains",
        "always", "never", "must", "should", "requires",
    }
    
    # Speculation/hedging (reduce importance)
    HEDGING_PHRASES = {
        "might", "may", "could", "seems", "appears", "possibly",
        "probably", "typically", "usually", "sometimes", "often",
        "arguably", "i think", "one could", "perhaps", "in my opinion",
    }
    
    @staticmethod
    def score_importance(text: str, source: str = "") -> float:
        """Score chunk importance (0.0-1.0).
        
        Higher = more important to store.
        - 0.8+: Core definitions, facts, foundational concepts
        - 0.6-0.8: Important supporting details
        - 0.4-0.6: Examples, explanations, context
        - <0.4: Casual remarks, speculation
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        score = 0.0
        
        # === BASE SCORE (content quality) ===
        # Length indicator (substantive content is longer)
        if word_count < 15:
            return 0.1  # Too short
        elif word_count < 50:
            score = 0.2
        elif word_count < 150:
            score = 0.4
        elif word_count < 300:
            score = 0.5
        else:
            score = 0.6  # Longer = more substantial
        
        # === BOOST: Content indicators ===
        
        # Technical specificity (many technical terms)
        tech_count = sum(1 for term in ImportanceScorer.TECHNICAL_TERMS 
                        if term in text_lower)
        if tech_count >= 3:
            score = min(1.0, score + 0.25)
        elif tech_count >= 1:
            score = min(1.0, score + 0.10)
        
        # Definitive statements (facts, not speculation)
        definitive_count = sum(1 for verb in ImportanceScorer.DEFINITIVE_VERBS 
                              if verb in text_lower)
        if definitive_count >= 3:
            score = min(1.0, score + 0.20)
        elif definitive_count >= 1:
            score = min(1.0, score + 0.10)
        
        # Specific facts (numbers, citations, dates)
        has_numbers = bool(re.search(r'\d+', text))
        has_percent = bool(re.search(r'\d+\s*%', text))
        has_list = bool(re.search(r'(?:^|\n)[\s]*[-*●]\s', text, re.MULTILINE))
        
        if has_numbers:
            score = min(1.0, score + 0.10)
        if has_percent:
            score = min(1.0, score + 0.15)
        if has_list:
            score = min(1.0, score + 0.15)
        
        # Source quality (web sources tend to be higher quality)
        if source.startswith("web:"):
            score = min(1.0, score + 0.05)
        
        # === PENALTY: Hedging/speculation ===
        hedging_count = sum(1 for phrase in ImportanceScorer.HEDGING_PHRASES 
                           if phrase in text_lower)
        
        if hedging_count >= 5:
            score = max(0.0, score - 0.30)
        elif hedging_count >= 2:
            score = max(0.0, score - 0.15)
        
        # Question marks indicate questions, not statements (lower priority)
        question_count = text.count('?')
        if question_count >= 2:
            score = max(0.0, score - 0.20)
        
        # === EXPLANATION BOOST ===
        # Chunks that explain/define concepts are valuable
        if re.search(r'(?:is|are|means|refers to|denotes|defines)\s+(?:a|an|the)?\s+\w+', text_lower):
            score = min(1.0, score + 0.15)
        
        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def should_store(text: str, source: str = "", threshold: float = 0.5) -> tuple[bool, float]:
        """Determine if chunk should be permanently stored.
        
        Args:
            text: Chunk text
            source: Content source
            threshold: Minimum importance to store (0.0-1.0)
        
        Returns:
            (should_store, importance_score)
        """
        score = ImportanceScorer.score_importance(text, source)
        return score >= threshold, score


__all__ = ["ImportanceScorer"]
