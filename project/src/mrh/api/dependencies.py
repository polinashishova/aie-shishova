""" src/mrh/api/dependencies.py
Зависимости для API приложения: проверка доступности моделей, загрузка моделей в память.
"""

from functools import lru_cache
from pathlib import Path
from fastapi import HTTPException, status
from typing import Optional

from sklearn.pipeline import Pipeline
from implicit.cpu.als import AlternatingLeastSquares
import logging
import numpy as np
from scipy.sparse import csr_matrix

from mrh.models import load_model
from mrh.data import load_data
from mrh.utils import load_json, inverse_dict
from mrh.config import CONFIGS_DIR


logger = logging.getLogger(__name__) 


def is_models_ready() -> bool:
    """
    Проверка готовности моделей через доступ к кэшу.
    """
    try:
        return (load_cb_model.cache_info().currsize > 0 and load_cf_model.cache_info().currsize > 0)
    except (AttributeError, TypeError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def get_config_paths() -> dict[str, str]:
    """
    Загрузка конфигурации путей к файлам.
    
    Возвращает
    -------
    dict[str, str]
        Словарь с путями к файлам и директориям из конфигурации
    """

    root_path = Path.cwd()
    return load_json(root_path / CONFIGS_DIR / 'paths.json')


@lru_cache(maxsize=1)
def load_cb_model() -> tuple[Pipeline, 
           np.ndarray,
           dict[int, int],
           dict[int, int]]:
    """
    Загрузка контент-базированной модели и сопутствующих данных.
    
    Возвращает
    -------
    tuple[Pipeline, np.ndarray, dict[int, int], dict[int, int]]
        Кортеж из четырёх элементов:
        - cb_pipeline : Pipeline
            Обученный пайплайн (TfidfVectorizer + TruncatedSVD)
        - cb_features : np.ndarray
            Матрица признаков фильмов (n_movies, n_components)
        - cb_movieId_to_idx : dict[int, int]
            Словарь отображения movieId -> индекс в матрице
        - cb_idx_to_movieId : dict[int, int]
            Словарь отображения индекс -> movieId
    
    Исключения
    ------
    HTTPException (503 SERVICE UNAVAILABLE)
        Если не удалось загрузить модель или сопутствующие файлы
    """

    root_path = Path.cwd()
    
    paths = get_config_paths()
    data_processed_dir = root_path / paths['data_processed_dir']
    data_features_dir = root_path / paths['data_features_dir']
    models_dir = root_path / paths['artifacts_dir'] / paths['models_dir']

    cb_pipeline_path = models_dir / paths['cb_pipeline']
    cb_features_path = data_features_dir / paths['cb_features']
    cb_movieId_to_idx_path = data_processed_dir / paths['cb_movieId_to_idx']
    

    try:
        logger.info('Загрузка контентной модели ...')
        cb_pipeline = load_model(cb_pipeline_path)
        cb_features = load_data(cb_features_path)
        if hasattr(cb_features, 'to_numpy'):
            features_array = cb_features.to_numpy()
        else:
            features_array = np.asarray(cb_features)
        cb_movieId_to_idx = {int(k): int(v) for k, v in load_json(cb_movieId_to_idx_path).items()}
        cb_idx_to_movieId = inverse_dict(cb_movieId_to_idx)

        logger.info('Контентная модель загружена %d фильмов, %d признаков',
                    len(cb_movieId_to_idx), cb_features.shape[1])
        
        return cb_pipeline, features_array, cb_movieId_to_idx, cb_idx_to_movieId
    
    except Exception as e:
        logger.exception('Ошибка загрузки контентной модели')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Не удалось загрузить контентную модель: {e}'
        )
    

@lru_cache(maxsize=1)
def load_cf_model() -> tuple[AlternatingLeastSquares, 
           csr_matrix, 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[Optional[int], Optional[list[int]]]]:
    """
    Загрузка коллаборативной модели ALS и сопутствующих данных.
    
    Возвращает
    -------
    tuple[AlternatingLeastSquares, csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[Optional[int], Optional[list[int]]]]
        Кортеж из семи элементов:
        - cf_model : AlternatingLeastSquares
            Обученная ALS модель
        - cf_user_item_matrix: csr_matrix
            Матрица обучающих взаимодействий пользователей и фильмов
        - cf_movieId_to_idx : dict[int, int]
            Словарь отображения movieId -> индекс в матрице
        - cf_userId_to_idx : dict[int, int]
            Словарь отображения userId -> индекс в матрице
        - cf_idx_to_movieId : dict[int, int]
            Словарь отображения индекс -> movieId
        - cf_idx_to_userId : dict[int, int]
            Словарь отображения индекс -> userId
        - cf_neg_movieIds: dict[Optional[int], Optional[list[int]]]
            Словарь фильмов, которые не нужно рекомендовать определенным пользователям
    
    Исключения
    ------
    HTTPException (503 SERVICE UNAVAILABLE)
        Если не удалось загрузить модель или сопутствующие файлы
    """

    root_path = Path.cwd()
    
    paths = get_config_paths()
    data_processed_dir = root_path / paths['data_processed_dir']
    models_dir = root_path / paths['artifacts_dir'] / paths['models_dir']

    cf_model_path = models_dir / paths['als_model']
    cf_user_item_path = data_processed_dir / paths['cf_user_item_matrix']
    cf_movieId_to_idx_path = data_processed_dir / paths['cf_movieId_to_idx']
    cf_userId_to_idx_path = data_processed_dir / paths['cf_userId_to_idx']
    cf_neg_movieIds_path = data_processed_dir / paths['neg_cf_movieIds']
    

    try:
        logger.info('Загрузка колаборативной модели ...')
        cf_model = load_model(cf_model_path)
        cf_user_item_matrix = load_data(cf_user_item_path)
        cf_movieId_to_idx = {int(k): int(v) for k, v in load_json(cf_movieId_to_idx_path).items()}
        cf_userId_to_idx = {int(k): int(v) for k, v in load_json(cf_userId_to_idx_path).items()}
        cf_idx_to_movieId = inverse_dict(cf_movieId_to_idx)
        cf_idx_to_userId = inverse_dict(cf_userId_to_idx)
        if cf_neg_movieIds_path.exists():
            cf_neg_movieIds = {int(k): v for k, v in load_json(cf_neg_movieIds_path).items()}
        else:
            cf_neg_movieIds = {}

        logger.info('Колаборативная модель загружена %d пользователей, %d фильмов',
                    len(cf_userId_to_idx), len(cf_movieId_to_idx))
        
        return cf_model, cf_user_item_matrix, cf_movieId_to_idx, cf_userId_to_idx, cf_idx_to_movieId, cf_idx_to_userId, cf_neg_movieIds
    
    except Exception as e:
        logger.exception('Ошибка загрузки колаборативной модели')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Не удалось загрузить колаборативную модель: {e}'
        )


def get_cb_model_dep() -> tuple[Pipeline, 
           np.ndarray, 
           dict[int, int], 
           dict[int, int]]:
    """
    Dependency для внедрения контент-базированной модели в FastAPI эндпоинты.
    
    Возвращает
    -------
    tuple[Pipeline, np.ndarray, dict[int, int], dict[int, int]]
        Данные контент-базированной модели (см. load_cb_model)
    """

    return load_cb_model()

def get_cf_model_dep() -> tuple[AlternatingLeastSquares, 
           csr_matrix, 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[Optional[int], Optional[list[int]]]]:
    """
    Dependency для внедрения коллаборативной модели в FastAPI эндпоинты.
    
    Возвращает
    -------
    tuple[AlternatingLeastSquares, csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[Optional[int], Optional[list[int]]]]
        Данные коллаборативной модели (см. load_cf_model)
    """

    return load_cf_model()