"""
Скрипт для скачивания необходимых для инференса моделей и данных.

Пример использования:
cd project
python scripts/download_models.py
"""

from huggingface_hub import snapshot_download
import os
import shutil
from pathlib import Path
import logging
from mrh.utils import load_json


REPO_ID = "shajeless/movie-recsys-hybrid"
ROOT_PATH = Path.cwd()
paths_path = ROOT_PATH / 'configs' / 'paths.json'
paths = load_json(paths_path)
MODEL_DIR = ROOT_PATH / paths["artifacts_dir"] / paths["models_dir"]
PROCESSED_DIR = ROOT_PATH / paths["data_processed_dir"]
FEATURES_DIR = ROOT_PATH / paths["data_features_dir"]

REQUIRED_FILES = {
    'models': ['cb_pipeline.joblib', 'als_model.joblib'],
    'processed': ['cf_user_item_matrix.npz', 'cf_movieId_to_idx.json', 
                  'cf_userId_to_idx.json', 'cb_movieId_to_idx.json', 
                  'neg_cf_movieIds.json'],
    'features': ['cb_features.csv']
}


def check_files_exist() -> tuple[bool, list[str]]:
    """
    Проверяет, существуют ли все необходимые файлы.
    
    Returns:
        tuple[bool, list[str]]: (все_ли_файлы_существуют, список_отсутствующих_файлов)
    """
    missing_files = []
    
    for file in REQUIRED_FILES['models']:
        if not (MODEL_DIR / file).exists():
            missing_files.append(f"models/{file}")
    
    for file in REQUIRED_FILES['processed']:
        if not (PROCESSED_DIR / file).exists():
            missing_files.append(f"data/processed/{file}")
    
    for file in REQUIRED_FILES['features']:
        if not (FEATURES_DIR / file).exists():
            missing_files.append(f"data/features/{file}")
    
    return len(missing_files) == 0, missing_files


def download_all() -> None:
    """
    Загружает все файлы из репозитория в нужные папки.
    Если все файлы уже существуют, загрузка пропускается.
    """
    
    all_exist, missing = check_files_exist()
    
    if all_exist:
        logger.info("Все необходимые файлы уже существуют. Загрузка пропущена.")
        logger.info("Файлы находятся в:")
        logger.info("  - Модели: %s", MODEL_DIR)
        logger.info("  - Processed данные: %s", PROCESSED_DIR)
        logger.info("  - Features данные: %s", FEATURES_DIR)
        return
    
    logger.info("Отсутствуют следующие файлы: %s", missing)
    logger.info("Начинаю загрузку...")
    
    temp_dir = ROOT_PATH / ".temp_download"
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info("Создана временная директория: %s", temp_dir)
        
        logger.info("Начало загрузки из репозитория %s", REPO_ID)
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=temp_dir
        )
        logger.info("Загрузка из репозитория успешно завершена")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        
        source_models = temp_dir / "models"
        if source_models.exists():
            for file in source_models.glob("*"):
                try:
                    shutil.copy2(file, MODEL_DIR / file.name)
                    logger.info("Скопирована модель: %s", file.name)
                except OSError as e:
                    logger.error("Не удалось скопировать модель %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Директория моделей не найдена: %s", source_models)
        
        source_processed = temp_dir / "data" / "processed"
        if source_processed.exists():
            for file in source_processed.glob("*"):
                try:
                    shutil.copy2(file, PROCESSED_DIR / file.name)
                    logger.info("Скопированы processed данные: %s", file.name)
                except OSError as e:
                    logger.error("Не удалось скопировать processed данные %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Директория processed данных не найдена: %s", source_processed)
        
        source_features = temp_dir / "data" / "features"
        if source_features.exists():
            for file in source_features.glob("*"):
                try:
                    shutil.copy2(file, FEATURES_DIR / file.name)
                    logger.info("Скопированы features данные: %s", file.name)
                except OSError as e:
                    logger.error("Не удалось скопировать features данные %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Директория features данных не найдена: %s", source_features)
        
        logger.info("Все модели и данные успешно загружены!")
        
    except Exception as e:
        logger.exception("Ошибка при загрузке моделей и данных")
        raise
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Временная директория удалена: %s", temp_dir)
            except OSError as e:
                logger.warning("Не удалось удалить временную директорию %s: %s", temp_dir, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    download_all()