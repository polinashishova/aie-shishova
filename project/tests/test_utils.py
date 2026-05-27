"""Тесты для утилитных функций."""

import pytest
import json
from pathlib import Path
import tempfile
from mrh.utils import load_json, save_json, inverse_dict, setup_logging


class TestInverseDict:
    """Тесты для функции inverse_dict."""
    
    def test_normal_dict(self):
        """Тест: обычный словарь."""
        d = {1: 'a', 2: 'b', 3: 'c'}
        expected = {'a': 1, 'b': 2, 'c': 3}
        assert inverse_dict(d) == expected
    
    def test_empty_dict(self):
        """Тест: пустой словарь."""
        assert inverse_dict({}) == {}
    
    def test_dict_with_duplicate_values(self):
        """Тест: словарь с дубликатами значений."""
        d = {1: 'a', 2: 'a', 3: 'b'}
        result = inverse_dict(d)
        assert len(result) == 2
        assert result['a'] == 2  # Последнее значение
        assert result['b'] == 3
    
    def test_invalid_input(self):
        """Тест: невалидный вход (не словарь)."""
        with pytest.raises(TypeError, match="Ожидается словарь"):
            inverse_dict([1, 2, 3])


class TestJsonOperations:
    """Тесты для работы с JSON."""
    
    def test_save_and_load_json(self):
        """Тест: сохранение и загрузка JSON."""
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            save_json(data, tmp_path)
            loaded_data = load_json(tmp_path)
            assert loaded_data == data
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Тест: загрузка несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            load_json(Path("/nonexistent/path/file.json"))
    
    def test_load_invalid_json(self):
        """Тест: загрузка невалидного JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("{invalid json}")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_json(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestSetupLogging:
    """Тесты для настройки логирования."""
    
    def test_setup_logging_creates_log_dir(self, tmp_path):
        """Тест: создается директория для логов."""
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        
        setup_logging(level="INFO", log_dir=log_dir, log_filename="test.log")
        
        assert log_dir.exists()
    
    def test_setup_logging_returns_logger(self, tmp_path):
        """Тест: функция возвращает логгер."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(level="INFO", log_dir=log_dir, log_filename="test.log")
        
        assert logger is not None
        assert logger.level == 20