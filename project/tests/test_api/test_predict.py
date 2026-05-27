"""Тесты для predict эндпоинта."""

import pytest
from fastapi.testclient import TestClient
from mrh.api.main import create_app


class TestPredictEndpoint:
    """Тесты для /predict эндпоинта."""
    
    @pytest.fixture
    def client(self):
        """Фикстура: тестовый клиент FastAPI."""
        return TestClient(create_app())
    
    
    def test_predict_without_ids(self, client):
        """Тест: запрос без movieIds и userIds."""
        response = client.post(
            "/predict",
            json={"k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_both_ids(self, client):
        """Тест: запрос с обоими типами ID."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "userIds": [1], "k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_empty_movieIds(self, client):
        """Тест: пустой список movieIds."""
        response = client.post(
            "/predict",
            json={"movieIds": [], "k": 10}
        )
        
        assert response.status_code == 400
        assert "не может быть пустым" in response.text
    
    def test_predict_with_empty_userIds(self, client):
        """Тест: пустой список userIds."""
        response = client.post(
            "/predict",
            json={"userIds": [], "k": 10}
        )
        
        assert response.status_code == 400
        assert "не может быть пустым" in response.text
    
    def test_predict_with_negative_ids(self, client):
        """Тест: отрицательные ID."""
        response = client.post(
            "/predict",
            json={"movieIds": [-1, -2], "k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_k_out_of_range(self, client):
        """Тест: k вне допустимого диапазона."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 200}
        )
        assert response.status_code == 422
        
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 0}
        )
        assert response.status_code == 422
    
    def test_predict_response_format(self, client):
        """Тест: формат ответа."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "model_used" in data
            assert "timestamp" in data
            assert "request_id" in data