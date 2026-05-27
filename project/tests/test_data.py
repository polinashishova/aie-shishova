"""Тесты для функций обработки данных."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import tempfile
from mrh.data import (
    preprocess_tags, 
    get_ratings_by_threshold,
    save_data,
    load_data
)


class TestPreprocessTags:
    """Тесты для предобработки тегов."""
    
    def test_normal_string(self):
        """Тест: обычная строка."""
        assert preprocess_tags("Hello, World!") == "hello world"
    
    def test_string_with_punctuation(self):
        """Тест: строка с пунктуацией."""
        assert preprocess_tags("Fantasy! Action? Drama...") == "fantasy action drama"
    
    def test_uppercase_string(self):
        """Тест: строка в верхнем регистре."""
        assert preprocess_tags("COMEDY ROMANCE") == "comedy romance"
    
    def test_empty_string(self):
        """Тест: пустая строка."""
        assert preprocess_tags("") == ""
    
    def test_nan_value(self):
        """Тест: NaN значение."""
        assert preprocess_tags(float('nan')) == ""
    
    def test_none_value(self):
        """Тест: None значение."""
        assert preprocess_tags(None) == ""
    
    def test_russian_letters(self):
        """Тест: русские буквы."""
        assert preprocess_tags("Привет, Мир! 123") == "привет мир 123"
    
    def test_numbers_only(self):
        """Тест: только цифры."""
        assert preprocess_tags("123 456") == "123 456"
    
    def test_invalid_type(self):
        """Тест: невалидный тип."""
        with pytest.raises(TypeError, match="должен быть str, float или None"):
            preprocess_tags(["list", "is", "invalid"])


class TestGetRatingsByThreshold:
    """Тесты для фильтрации рейтингов по порогу."""
    
    @pytest.fixture
    def sample_ratings(self):
        """Фикстура: пример данных с рейтингами."""
        return pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [101, 102, 201, 202, 301, 302],
            'rating': [2.5, 4.0, 3.5, 5.0, 1.5, 4.5],
            'timestamp': [1000, 1001, 2000, 2001, 3000, 3001]
        })
    
    def test_split_at_threshold(self, sample_ratings):
        """Тест: разделение по порогу."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=4.0)
        
        assert len(neg) == 3
        assert all(rating < 4.0 for rating in neg['rating'])
        
        assert len(pos) == 3
        assert all(rating >= 4.0 for rating in pos['rating'])
    
    def test_threshold_3_5(self, sample_ratings):
        """Тест: порог 3.5."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=3.5)
        
        assert len(neg) == 2
        assert len(pos) == 4
    
    def test_threshold_at_min(self, sample_ratings):
        """Тест: порог на минимальном значении."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=1.5)
        
        assert len(neg) == 0  
        assert len(pos) == 6   
    
    def test_threshold_at_max(self, sample_ratings):
        """Тест: порог на максимальном значении."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=5.0)
        
        assert len(neg) == 5 
        assert len(pos) == 1
    
    def test_invalid_ratings_type(self):
        """Тест: ratings не DataFrame."""
        with pytest.raises(TypeError):
            get_ratings_by_threshold([1, 2, 3], 4.0)
    
    def test_missing_columns(self):
        """Тест: отсутствуют необходимые колонки."""
        df = pd.DataFrame({'wrong_col': [1, 2]})
        with pytest.raises(ValueError, match="отсутствуют колонки"):
            get_ratings_by_threshold(df, 4.0)
    
    def test_empty_dataframe(self):
        """Тест: пустой DataFrame."""
        df = pd.DataFrame(columns=['movieId', 'rating'])
        with pytest.raises(ValueError, match="пуст"):
            get_ratings_by_threshold(df, 4.0)


class TestSaveLoadData:
    """Тесты для сохранения и загрузки данных."""
    
    def test_save_and_load_dataframe(self, tmp_path):
        """Тест: сохранение и загрузка DataFrame."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        file_path = tmp_path / "test_df.csv"
        
        save_data(file_path, df)
        assert file_path.exists()
        
        loaded = load_data(file_path)
        pd.testing.assert_frame_equal(df, loaded)
    
    def test_save_and_load_numpy_array(self, tmp_path):
        """Тест: сохранение и загрузка numpy массива."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        file_path = tmp_path / "test_array.csv"
        
        save_data(file_path, arr)
        assert file_path.exists()
        
        loaded = load_data(file_path)
        assert isinstance(loaded, pd.DataFrame)
        np.testing.assert_array_equal(arr, loaded.values)
    
    def test_save_existing_file_skip(self, tmp_path, caplog):
        """Тест: существующий файл пропускается."""
        caplog.set_level(logging.INFO)
        df = pd.DataFrame({'col': [1, 2]})
        file_path = tmp_path / "existing.csv"

        save_data(file_path, df)
        assert file_path.exists()
        
        save_data(file_path, df)
        
        assert "уже существует" in caplog.text
    
    def test_load_nonexistent_file(self, tmp_path):
        """Тест: загрузка несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.npz")
    
    def test_load_unsupported_format(self, tmp_path):
        """Тест: загрузка неподдерживаемого формата."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("some text")
        
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            load_data(file_path)