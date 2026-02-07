"""Integration tests for REST API."""

import pytest
import json
from unittest.mock import MagicMock, patch
from Zypherus.api.server import ZypherusAPIServer, APIResponse


class MockACE:
    """Mock ACE system for testing."""
    
    def __init__(self):
        self.memory = MagicMock()
        self.memory.entries = [
            {"text": "Python is a language", "source": "test"},
            {"text": "Ruby is also a language", "source": "test"}
        ]
        self.claim_store = MagicMock()
        self.claim_store.claims = {
            "claim1": {
                "subject": "Python",
                "predicate": "is_a",
                "object": "language",
                "confidence": 0.95
            }
        }
        self.concept_graph = MagicMock()
        self.concept_graph.nodes = {"Python": {}, "language": {}}
        self.concept_graph.edges = {}
    
    def ingest_document(self, source: str, text: str):
        """Mock ingest."""
        self.memory.entries.append({"text": text, "source": source})
    
    def answer(self, query: str):
        """Mock answer."""
        return {
            "answer": f"Answer to: {query}",
            "confidence": 0.8,
            "sources": []
        }


@pytest.fixture
def ace():
    """Create mock ACE instance."""
    return MockACE()


@pytest.fixture
def client(ace):
    """Create Flask test client."""
    server = ZypherusAPIServer(ace)
    app = server.create_flask_app()
    app.config["TESTING"] = True
    
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "status" in data["data"]


class TestIngestEndpoint:
    """Test ingestion endpoint."""
    
    def test_ingest_success(self, client):
        """Test successful ingestion."""
        payload = {
            "text": "Python is amazing",
            "source": "test"
        }
        response = client.post(
            "/api/ingest",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
    
    def test_ingest_empty_text(self, client):
        """Test ingestion with empty text."""
        payload = {
            "text": "",
            "source": "test"
        }
        response = client.post(
            "/api/ingest",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False


class TestAnswerEndpoint:
    """Test question answering endpoint."""
    
    def test_answer_success(self, client):
        """Test successful answer."""
        payload = {
            "query": "What is Python?"
        }
        response = client.post(
            "/api/answer",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "answer" in data["data"]
    
    def test_answer_empty_query(self, client):
        """Test answer with empty query."""
        payload = {
            "query": ""
        }
        response = client.post(
            "/api/answer",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400


class TestStatusEndpoint:
    """Test status endpoint."""
    
    def test_get_status(self, client):
        """Test get status."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "uptime" in data["data"]


class TestMemoryEndpoint:
    """Test memory endpoint."""
    
    def test_get_memory(self, client):
        """Test get memory."""
        response = client.get("/api/memory")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["total_entries"] >= 0


class TestSearchEndpoint:
    """Test search endpoint."""
    
    def test_search_success(self, client):
        """Test successful search."""
        payload = {
            "query": "Python",
            "limit": 10
        }
        response = client.post(
            "/api/search",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "results" in data["data"]
    
    def test_search_empty_query(self, client):
        """Test search with empty query."""
        payload = {
            "query": ""
        }
        response = client.post(
            "/api/search",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400


class TestConceptsEndpoint:
    """Test concepts endpoint."""
    
    def test_get_concepts(self, client):
        """Test get concepts."""
        response = client.get("/api/concepts")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


class TestBeliefsEndpoint:
    """Test beliefs endpoint."""
    
    def test_get_beliefs(self, client):
        """Test get beliefs."""
        response = client.get("/api/beliefs")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


class TestAPIDocumentation:
    """Test API documentation."""
    
    def test_get_docs(self, client):
        """Test get API docs."""
        response = client.get("/api/docs")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "endpoints" in data
        assert "authentication" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
