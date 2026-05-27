"""
Скрипт для скачивания, распаковки, подготовки и сохранения данных для обучения моделей рекомендаций.

Пример использования:
cd project
python scripts/load_and_prepare_data.py
"""

from mrh.data import download_data, extract_archive, feature_preparation_cb, get_ratings_by_threshold, data_preparation_cf, save_data
from pathlib import Path
from mrh.utils import setup_logging, load_json, save_json
from mrh.config import CB_THRESHOLD, CF_THRESHOLD, CONFIGS_DIR
import pandas as pd


def load_and_prepare_data(cb_thr: float = 3.0, cf_thr: float = 4.0) -> None:
    """
    Загрузка, очистка и подготовка данных MovieLens 32M для гибридной системы рекомендаций.
    
    Параметры
    ----------
    cb_thr : float, default=3.0
        Порог рейтинга для контент-базированной модели (фильмы с рейтингом >= порога)
    cf_thr : float, default=4.0
        Порог рейтинга для коллаборативной модели (позитивные взаимодействия)
    
    Возвращает
    -------
    None
        Данные сохраняются в файлы согласно конфигурации
    
    Исключения
    ----------
    ValueError
        Если после фильтрации не осталось данных
    FileNotFoundError, OSError
        Если не удалось загрузить или сохранить файлы
    """

    root_path = Path(__file__).resolve().parent.parent

    paths_path = root_path / CONFIGS_DIR / "paths.json"
    paths = load_json(paths_path)
    data_url = paths['data_url']
    data_raw_dir = root_path / paths['data_raw_dir']
    ml_32m_zip = data_raw_dir / paths['ml_32m_zip']
    download_data(url=data_url, path=ml_32m_zip)
    ml_32m = data_raw_dir / paths['ml_32m']
    extract_archive(path_from=ml_32m_zip, directory=data_raw_dir, expected_path=ml_32m)

    ratings = pd.read_csv(ml_32m / paths["ratings_path"])
    movies = pd.read_csv(ml_32m / paths["movies_path"])
    tags = pd.read_csv(ml_32m / paths["tags_path"])

    assert not ratings.empty, "Файл ratings пуст или не прочитан"
    assert not movies.empty, "Файл movies пуст или не прочитан"

    _, cb_ratings = get_ratings_by_threshold(ratings, threshold=cb_thr)
    neg_cf_ratings, cf_ratings = get_ratings_by_threshold(ratings, threshold=cf_thr)

    if cb_ratings.empty:
        raise ValueError(f"После фильтрации по порогу {cb_thr} не осталось данных для CB-модели")
    if cf_ratings.empty:
        raise ValueError(f"После фильтрации по порогу {cf_thr} не осталось данных для CF-модели")

    neg_cf_ratings_groups = neg_cf_ratings.groupby('userId')
    neg_movieIds = {int(userId): [int(movieId) for movieId in group['movieId'].to_list()]
                    for userId, group in neg_cf_ratings_groups}

    cb_prefeatures = feature_preparation_cb(movies, tags, cb_ratings)
    cf_data = data_preparation_cf(cf_ratings)

    cb_unique_movieIds = cb_ratings['movieId'].unique()
    cf_unique_movieIds = cf_ratings['movieId'].unique()
    cf_unique_userIds = cf_ratings['userId'].unique()

    cb_movieId_to_idx = {int(mid): i for i, mid in enumerate(cb_unique_movieIds)}
    cf_movieId_to_idx = {int(mid): i for i, mid in enumerate(cf_unique_movieIds)}
    cf_userId_to_idx = {int(uid): i for i, uid in enumerate(cf_unique_userIds)}

    data_processed_dir = root_path / paths['data_processed_dir']
    processed_cb_movies_path = data_processed_dir / paths['movies_processed']
    save_data(processed_cb_movies_path, cb_prefeatures)

    cf_data_path = data_processed_dir / paths['cf_user_item_matrix']
    save_data(cf_data_path, cf_data)

    save_json(cb_movieId_to_idx, data_processed_dir / paths['cb_movieId_to_idx'])
    save_json(cf_movieId_to_idx, data_processed_dir / paths['cf_movieId_to_idx'])
    save_json(cf_userId_to_idx, data_processed_dir / paths['cf_userId_to_idx'])
    save_json(neg_movieIds, data_processed_dir / paths['neg_cf_movieIds'])


if __name__ == '__main__':
    logger = setup_logging()
    logger.info("Начало загрузки и подготовки данных")
    try:
        load_and_prepare_data(cb_thr=CB_THRESHOLD, cf_thr=CF_THRESHOLD)
        logger.info("Успешное завершение загрузки и подготовки данных")
    except Exception as e:
        logger.exception("Критическая ошибка в процессе загрузки и подготовки данных")
        raise
