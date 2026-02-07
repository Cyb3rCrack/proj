"""Tests for API request/response schemas."""

import pytest
from Zypherus.api.schemas import (
    IngestRequest, QuestionRequest, SearchRequest, 
    BeliefQuery, MemoryRequest, SourceEnum, MemoryAction
)


class TestIngestRequest:
    """Test ingest request schema."""
    
    def test_valid_ingest_request(self):
        """Test valid ingest request."""
        data = {
            "source_type": "text",
            "content": "Python is a programming language"
        }
        request = IngestRequest(**data)  # type: ignore[arg-type]
        assert request.source_type == SourceEnum.TEXT
        assert request.content == "Python is a programming language"
    
    def test_ingest_request_empty_content(self):
        """Test ingest request with empty content."""
        data = {
            "source_type": "text",
            "content": ""
        }
        with pytest.raises(ValueError):
            IngestRequest(**data)  # type: ignore[arg-type]
    
    def test_ingest_request_with_metadata(self):
        """Test ingest request with metadata."""
        data = {
            "source_type": "url",
            "content": "Some content",
            "metadata": {"author": "test", "date": "2026-02-06"}
        }
        request = IngestRequest(**data)  # type: ignore[arg-type]
        assert request.metadata and request.metadata["author"] == "test"


class TestQuestionRequest:
    """Test question request schema."""
    
    def test_valid_question_request(self):
        """Test valid question request."""
        data = {
            "query": "What is Python?"
        }
        request = QuestionRequest(**data)  # type: ignore[arg-type]
        assert request.query == "What is Python?"
    
    def test_question_with_all_fields(self):
        """Test question with all fields."""
        data = {
            "query": "What is Python?",
            "include_sources": True,
            "max_results": 5,
            "confidence_threshold": 0.7
        }
        request = QuestionRequest(**data)
        assert request.max_results == 5
        assert request.confidence_threshold == 0.7
    
    def test_question_max_results_validation(self):
        """Test max results validation."""
        data = {
            "query": "test",
            "max_results": 500  # Exceeds max of 20
        }
        with pytest.raises(ValueError):
            QuestionRequest(**data)


class TestSearchRequest:
    """Test search request schema."""
    
    def test_valid_search_request(self):
        """Test valid search request."""
        data = {
            "query": "machine learning",
            "search_type": "semantic",
            "limit": 10
        }
        request = SearchRequest(**data)
        assert request.query == "machine learning"


class TestBeliefQuery:
    """Test belief query schema."""
    
    def test_valid_belief_query(self):
        """Test valid belief query."""
        data = {
            "entity": "Python",
            "include_relationships": True
        }
        query = BeliefQuery(**data)
        assert query.entity == "Python"


class TestMemoryRequest:
    """Test memory request schema."""
    
    def test_get_stats_action(self):
        """Test get stats action."""
        data = {
            "action": "get_stats"
        }
        request = MemoryRequest(**data)  # type: ignore[arg-type]
        assert request.action == MemoryAction.GET_STATS
    
    def test_memory_action_validation(self):
        """Test memory action validation."""
        data = {
            "action": "invalid_action"
        }
        with pytest.raises(ValueError):
            MemoryRequest(**data)  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
