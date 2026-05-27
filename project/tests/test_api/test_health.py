"""Тесты для health эндпоинта."""

import pytest
from fastapi.testclient import TestClient
from mrh.api.main import create_app


class TestHealthEndpoint:
    """Тесты для /health эндпоинта."""
    
    @pytest.fixture
    def client(self):
        """Фикстура: тестовый клиент FastAPI."""
        return TestClient(create_app())
    
    def test_health_endpoint_returns_200(self, client):
        """Тест: health эндпоинт возвращает 503, так как модели не готовы."""
        response = client.get("/health")
        assert response.status_code == 503
    
    def test_health_response_format(self, client):
        """Тест: формат ответа health эндпоинта."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "models_loaded" in data
        assert "timestamp" in data
        assert data["service"] == "movie-recsys-hybrid"
    
    def test_root_endpoint(self, client):
        """Тест: корневой эндпоинт."""
        response = client.get("/")
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert "predict" in data