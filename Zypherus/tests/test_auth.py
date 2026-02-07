"""Tests for Authentication system."""

import pytest
from Zypherus.api.auth import AuthConfig, TokenManager, AuthService


class TestTokenManager:
    """Test JWT token management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = AuthConfig()
        self.manager = TokenManager(self.config)
    
    def test_generate_token(self):
        """Test token generation."""
        token = self.manager.generate_token("user1", "admin")
        assert token is not None
        assert len(token) > 0
    
    def test_verify_valid_token(self):
        """Test verification of valid token."""
        token = self.manager.generate_token("user1", "admin")
        payload = self.manager.verify_token(token)
        
        assert payload is not None
        assert payload["user_id"] == "user1"
        assert payload["role"] == "admin"
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token."""
        payload = self.manager.verify_token("invalid.token.here")
        assert payload is None
    
    def test_api_key_verification(self):
        """Test API key verification."""
        # Set test API keys
        self.config.api_keys = {"test_key_123": {"role": "user"}}
        
        user = self.manager.verify_api_key("test_key_123")
        assert user is not None
        assert user["role"] == "user"
    
    def test_invalid_api_key(self):
        """Test invalid API key."""
        user = self.manager.verify_api_key("invalid_key")
        assert user is None


class TestAuthService:
    """Test authentication service."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.auth = AuthService()
    
    def test_auth_service_creation(self):
        """Test auth service can be created."""
        assert self.auth is not None
        assert self.auth.token_manager is not None
        assert self.auth.config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
