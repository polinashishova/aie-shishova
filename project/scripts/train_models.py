"""
Скрипт для обучения и сохранения рекомендательных моделей; сохранение признаков фильмов для content-based модели.

Пример использования:
cd project
python scripts/train_models.py
"""

from mrh.models import build_cb_pipeline, train_cb_pipeline, apply_fitted_cb_pipeline, build_als_estimator, train_als_estimator, save_model
from mrh.data import load_data, save_data
from mrh.utils import setup_logging, load_json
from mrh.config import CB_FEATURE_COLUMN, CONFIGS_DIR, LOG_LEVEL
from pathlib import Path
from typing import Optional


def train_and_save_models(cb_feature_column: Optional[str] = None) -> None:
    """
    Обучение и сохранение контент-базированной и коллаборативной моделей.
    
    Параметры
    ----------
    cb_feature_column : str, optional
        Название колонки в processed_movies для обучения CB-пайплайна.
        Если None, используется значение по умолчанию из конфигурации.
    
    Возвращает
    -------
    None
        Модели сохраняются в файлы согласно конфигурации
    
    Исключения
    ----------
    FileNotFoundError, KeyError
        Если конфигурационные файлы или данные не найдены
    ValueError, RuntimeError
        Если обучение или сохранение моделей не удалось
    """

    CONFIG_PATHS = CONFIGS_DIR + "/paths.json"
    CONFIG_MODELS = CONFIGS_DIR + "/models.json"
        
    root_path = Path(__file__).resolve().parent.parent

    paths_path = root_path / CONFIG_PATHS
    paths = load_json(paths_path)

    models_config_path = root_path / CONFIG_MODELS
    models_config = load_json(models_config_path)

    if 'cf' not in models_config or 'als' not in models_config['cf']:
        raise ValueError(
            "Конфигурация models.json должна содержать раздел 'cf' с параметрами 'als'. "
            f"Доступные ключи: {list(models_config.keys())}"
        )

    models_dir = root_path / paths['artifacts_dir'] / paths['models_dir']
    cb_pipeline_path = models_dir / paths['cb_pipeline']

    data_features_dir = root_path / paths["data_features_dir"]
    cb_features_path = data_features_dir / paths["cb_features"]

    cf_als_model_path = models_dir / paths['als_model']

    data_processed_dir = root_path / paths['data_processed_dir']


    if not (cb_pipeline_path.exists() and cb_features_path.exists()):

        processed_cb_movies_path = data_processed_dir / paths['movies_processed']
        processed_cb_movies = load_data(processed_cb_movies_path)

        cb_pipeline = build_cb_pipeline(models_config["cb"])
        cb_pipeline = train_cb_pipeline(cb_pipeline, processed_cb_movies, column=cb_feature_column)
        cb_features = apply_fitted_cb_pipeline(cb_pipeline, processed_cb_movies, column=cb_feature_column)

        save_data(cb_features_path, cb_features)
        save_model(cb_pipeline, cb_pipeline_path)

    
    if not cf_als_model_path.exists():

        cf_user_item_path = data_processed_dir / paths['cf_user_item_matrix']
        cf_user_item_matrix = load_data(cf_user_item_path)
        
        cf_als_model = build_als_estimator(models_config['cf']['als'])
        cf_als_model = train_als_estimator(cf_als_model, cf_user_item_matrix)

        save_model(cf_als_model, cf_als_model_path)


if __name__ == '__main__':
    logger = setup_logging(level=LOG_LEVEL)

    logger.info("Начало загрузки данных и обучения моделей")
    try:
        train_and_save_models(cb_feature_column=CB_FEATURE_COLUMN)
        logger.info("Успешное обучение и сохранения моделей")
    except Exception as e:
        logger.exception("Критическая ошибка в процессе обучения моделей")
        raise