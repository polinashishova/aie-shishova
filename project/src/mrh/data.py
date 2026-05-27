""" src/mrh/data.py
Работа с данными: скачивание, распаковка, подготовка и предобработка, сохранение и загрузка.
"""

import logging
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.nearest_neighbours import bm25_weight

logger = logging.getLogger(__name__)


def download_data(url: str, path: Path) -> None:
    """
    Скачивание файла по URL и сохранение по указанному пути.
    
    Параметры
    ----------
    url : str
        URL адрес файла для скачивания
    path : Path
        Полный путь с именем файла, куда будет сохранён файл
    
    Исключения
    ------
    urllib.error.URLError
        Если не удалось скачать файл
    OSError
        Если не удалось сохранить файл
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info('Файл уже существует по пути %s, пропуск скачивания', path)
        return

    try:
        logger.info('Скачивание %s в %s', url, path)
        urllib.request.urlretrieve(url, path)
        logger.info('Скачивание завершено: %s', path)
    except urllib.error.URLError:
        logger.exception('Не удалось скачать с %s', url)
        raise
    except OSError:
        logger.exception('Не удалось сохранить файл %s', path)
        raise


def extract_archive(
    path_from: Path, 
    directory: Path, 
    expected_path: Path
) -> None:
    """
    Распаковка zip архива из path_from в directory, если ожидаемый файл ещё не существует.
    
    Параметры
    ----------
    path_from : Path
        Путь к zip архиву
    directory : Path
        Директория для распаковки файлов
    expected_path : Path
        Путь, который должен существовать после успешной распаковки
    
    Исключения
    ------
    ValueError
        Если есть небезопсный путь в архиве
    """
    directory.mkdir(parents=True, exist_ok=True)

    if expected_path.exists():
        logger.info('Архив уже распакован, найден %s. Пропуск распаковки', expected_path)
        return

    with zipfile.ZipFile(path_from, 'r') as zip_ref:
        for member in zip_ref.namelist():
            member_path = (directory / member).resolve()
            if not str(member_path).startswith(str(directory.resolve())):
                raise ValueError(f"Небезопасный путь в архиве: {member}")
        zip_ref.extractall(path=directory)
    
    logger.info('Распаковка завершена: %s', directory)


def preprocess_tags(tags_string: str | float | None) -> str:
    """
    Предобработка строки тегов: приведение к нижнему регистру, удаление пунктуации и лишних пробелов.
    
    Параметры
    ----------
    tags_string : str, float, or None
        Исходная строка с тегами. Может быть:
        - str: обычная строка с тегами
        - float: обычно pd.NA или np.nan
        - None: значение None
    
    Возвращает
    -------
    str
        Обработанная строка тегов в нижнем регистре, содержащая только буквы (латиница,
        кириллица), цифры и пробелы. Лишние пробелы удалены.
        Если входная строка пуста или NaN, возвращается пустая строка.
    
    Исключения
    ------
    TypeError
        Если входной параметр имеет неподдерживаемый тип
    """
    if not isinstance(tags_string, (str, float, type(None))):
        raise TypeError(
            f"tags_string должен быть str, float или None, получен {type(tags_string).__name__}"
        )
    
    if pd.isna(tags_string) or tags_string == '' or tags_string is None:
        return ''
    
    if isinstance(tags_string, float):
        tags_string = str(tags_string)
    
    tags = tags_string.lower()
    tags = re.sub(r'[^a-zа-я0-9\s]', '', tags)
    tags = ' '.join(tags.split())
    
    return tags


def feature_preparation_cb(
    movies: pd.DataFrame, 
    tags: pd.DataFrame, 
    ratings: pd.DataFrame
) -> pd.DataFrame:
    """
    Подготовка признаков фильмов для контент-базированной модели.
    
    Функция создаёт текстовые признаки для фильмов на основе жанров, года выпуска
    и пользовательских тегов. Результат используется для построения TF-IDF матрицы
    и последующего поиска похожих фильмов.
    
    Параметры
    ----------
    movies : pd.DataFrame
        Датафрейм с фильмами, должен содержать колонки 'movieId', 'title', 'genres'
    tags : pd.DataFrame
        Датафрейм с тегами, должен содержать колонки 'movieId', 'tag'
    ratings : pd.DataFrame
        Датафрейм с рейтингами, должен содержать колонку 'movieId'.
        Только фильмы, присутствующие в этом датафрейме, будут включены в результат.
    
    Возвращает
    -------
    pd.DataFrame
        Обработанный датафрейм с колонками:
        - year: год выпуска фильма (int, 0 если год не определён)
        - genres_fixed: жанры в формате строки через пробелы (нижний регистр)
        - decade: десятилетие (например, '1990s', пустая строка если год = 0)
        - tags: оригинальные теги в виде строки (с дубликатами)
        - tags_clean: обработанные теги (с сохранением дубликатов)
        - genres_decade: комбинация жанров и десятилетия
        - genres_decade_tags: комбинация жанров, десятилетия и тегов
        
        Индекс датафрейма - movieId.
    
    Исключения
    ------
    TypeError
        Если movies не является pandas.DataFrame
        Если tags не является pandas.DataFrame
        Если ratings не является pandas.DataFrame
    ValueError
        Если в movies отсутствуют колонки 'movieId', 'title' или 'genres'
        Если в tags отсутствуют колонки 'movieId' или 'tag'
        Если в ratings отсутствует колонка 'movieId'
        Если датафрейм movies пуст
        Если датафрейм ratings пуст
        Если после фильтрации не осталось фильмов
    """
    if not isinstance(movies, pd.DataFrame):
        raise TypeError(f"movies должен быть pandas.DataFrame, получен {type(movies).__name__}")
    
    if not isinstance(tags, pd.DataFrame):
        raise TypeError(f"tags должен быть pandas.DataFrame, получен {type(tags).__name__}")
    
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings должен быть pandas.DataFrame, получен {type(ratings).__name__}")
    
    required_movie_cols = {'movieId', 'title', 'genres'}
    missing_movie_cols = required_movie_cols - set(movies.columns)
    if missing_movie_cols:
        raise ValueError(f"В movies отсутствуют колонки: {missing_movie_cols}")
    
    if 'movieId' not in tags.columns:
        raise ValueError("В tags отсутствует колонка 'movieId'")
    if 'tag' not in tags.columns:
        raise ValueError("В tags отсутствует колонка 'tag'")
    
    if 'movieId' not in ratings.columns:
        raise ValueError("В ratings отсутствует колонка 'movieId'")
    
    if len(movies) == 0:
        raise ValueError("Датафрейм movies пуст")
    
    if len(ratings) == 0:
        raise ValueError("Датафрейм ratings пуст")
    
    rated_movieIds = ratings['movieId'].unique().tolist()
    
    if len(rated_movieIds) == 0:
        raise ValueError("В ratings нет уникальных movieId")
    
    rated_movies = movies[movies['movieId'].isin(rated_movieIds)].copy()
    
    if len(rated_movies) == 0:
        raise ValueError(
            "Нет фильмов из ratings в датафрейме movies. "
            "Проверьте, что movieId из ratings существуют в movies"
        )
    
    logger.info(
        'Фильтрация фильмов: было %d, осталось %d (только оценённые пользователями)',
        len(movies), len(rated_movies)
    )
    
    processed_movies = rated_movies.copy()
    processed_movies.set_index('movieId', inplace=True)
    
    processed_movies['year'] = processed_movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    processed_movies['year'] = pd.to_numeric(processed_movies['year'], errors='coerce')
    processed_movies['year'] = processed_movies['year'].fillna(0).astype(int)

    processed_movies['genres_fixed'] = processed_movies['genres'].str.replace('|', ' ', regex=False).str.lower().str.strip()

    processed_movies['decade'] = processed_movies['year'].apply(lambda y: f"{(y // 10) * 10}s" if y > 0 else '')

    if len(tags) > 0:
        tags_clean = tags.copy()
        tags_clean['tag'] = tags_clean['tag'].fillna('')

        tags_filtered = tags_clean[tags_clean['movieId'].isin(rated_movieIds)]
        
        if len(tags_filtered) > 0:
            tags_grouped = tags_filtered.groupby('movieId')['tag'].agg(lambda x: ' '.join(x)).to_dict()
            processed_movies['tags'] = processed_movies.index.map(tags_grouped).fillna('')
        else:
            processed_movies['tags'] = ''
            logger.warning('Нет тегов для оценённых фильмов')
    else:
        processed_movies['tags'] = ''
        logger.warning('Датафрейм tags пуст')
    
    processed_movies['tags_clean'] = processed_movies['tags'].apply(preprocess_tags)

    processed_movies['genres_decade'] = (processed_movies['genres_fixed'] + ' ' + processed_movies['decade']).str.strip()
    
    processed_movies['genres_decade_tags'] = (processed_movies['genres_decade'] + ' ' + processed_movies['tags_clean']).str.strip()
    
    tags_non_empty = (processed_movies['tags_clean'] != '').sum()
    logger.info(
        'Признаки подготовлены: %d фильмов, из них %d имеют теги, средняя длина тегов %.1f символов',
        len(processed_movies),
        tags_non_empty,
        processed_movies['tags_clean'].str.len().mean() if tags_non_empty > 0 else 0
    )
    
    return processed_movies


def get_ratings_by_threshold(
    ratings: pd.DataFrame, 
    threshold: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Фильтрация рейтингов по пороговому значению. Делит DataFrame на два DataFrame
    по порогу рейтинга.
    
    Параметры
    ----------
    ratings : pd.DataFrame
        Датафрейм с рейтингами, должен содержать колонки 'movieId' и 'rating'
    threshold : float, default=3.0
        Пороговое значение рейтинга.
    
    Возвращает
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Кортеж отфильтрованных датафреймов с теми же колонками, что и входной.
        Первый со значениями, меньше порога, второй - больше или равными порогу.
    
    Исключения
    ------
    TypeError
        Если ratings не является pandas.DataFrame
        Если threshold не является числом (int или float)
    ValueError
        Если в ratings отсутствуют необходимые колонки 'movieId' или 'rating'
        Если датафрейм ratings пуст
    """
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings должен быть pandas.DataFrame, получен {type(ratings).__name__}")
    
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold должен быть int или float, получен {type(threshold).__name__}")
    
    required_cols = {'movieId', 'rating'}
    missing_cols = required_cols - set(ratings.columns)
    if missing_cols:
        raise ValueError(f"В ratings отсутствуют колонки: {missing_cols}")
    
    if len(ratings) == 0:
        raise ValueError("Датафрейм ratings пуст")
    
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()

    if not (min_rating <= threshold <= max_rating):
        logger.warning(
            'Порог %.1f выходит за пределы значений рейтинга в данных [%.1f, %.1f]',
            threshold, min_rating, max_rating
        )
    

    mask = ratings['rating'] >= threshold

    pos_ratings = ratings[mask].copy()
    neg_ratings = ratings[~mask].copy()
    
    logger.info(
        'Отфильтровано %d и %d записей с порогом %.1f (всего записей: %d)',
        len(neg_ratings),
        len(pos_ratings),
        threshold,
        len(ratings)
    )
    
    return (neg_ratings, pos_ratings)


def data_preparation_cf(
    ratings: pd.DataFrame, 
    K1: float = 100, 
    B: float = 0.8
) -> sparse.csr_matrix:
    """
    Подготовка данных для коллаборативной фильтрации с BM25 взвешиванием.
    
    Функция создаёт матрицу пользователь-фильм, преобразуя идентификаторы в индексы,
    и применяет BM25 взвешивание для нормализации весов взаимодействий.
    
    Параметры
    ----------
    ratings : pd.DataFrame
        Датафрейм с рейтингами, должен содержать колонки 'userId', 'movieId'
    K1 : float, default=100
        Параметр K1 для BM25, контролирует насыщение веса при повторных взаимодействиях
    B : float, default=0.8
        Параметр B для BM25, контролирует нормализацию длины (от 0 до 1)
    
    Возвращает
    -------
    sparse.csr_matrix
        Разреженная матрица пользователь-фильм в формате CSR с BM25 весами
        Размерность: (n_users, n_movies)
    
    Исключения
    ------
    TypeError
        Если ratings не является pandas.DataFrame
        Если K1 или B не являются числами
    ValueError
        Если в ratings отсутствуют колонки 'userId' или 'movieId'
        Если датафрейм ratings пуст
        Если после преобразования не осталось пользователей или фильмов
        Если K1 <= 0 или B не в диапазоне [0, 1]
    """
    
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings должен быть pandas.DataFrame, получен {type(ratings).__name__}")
    
    if not isinstance(K1, (int, float)):
        raise TypeError(f"K1 должен быть int или float, получен {type(K1).__name__}")
    
    if not isinstance(B, (int, float)):
        raise TypeError(f"B должен быть int или float, получен {type(B).__name__}")
    
    if K1 <= 0:
        raise ValueError(f"K1 должен быть больше 0, получен {K1}")
    
    if not (0 <= B <= 1):
        raise ValueError(f"B должен быть в диапазоне [0, 1], получен {B}")
    
    required_cols = {'userId', 'movieId'}
    missing_cols = required_cols - set(ratings.columns)
    if missing_cols:
        raise ValueError(f"В ratings отсутствуют колонки: {missing_cols}")
    
    if len(ratings) == 0:
        raise ValueError("Датафрейм ratings пуст")
    
    unique_movies = ratings['movieId'].unique()
    unique_users = ratings['userId'].unique()
    
    if len(unique_movies) == 0:
        raise ValueError("Нет уникальных movieId в ratings")
    
    if len(unique_users) == 0:
        raise ValueError("Нет уникальных userId в ratings")
    
    movieId_to_idx = {int(mid): i for i, mid in enumerate(unique_movies)}
    userId_to_idx = {int(uid): i for i, uid in enumerate(unique_users)}
    
    n_users = len(userId_to_idx)
    n_movies = len(movieId_to_idx)
    
    
    rows = [userId_to_idx[uid] for uid in ratings['userId']]
    cols = [movieId_to_idx[mid] for mid in ratings['movieId']]
    data = np.ones(len(ratings))
    
    user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    weighted_matrix = sparse.csr_matrix(bm25_weight(user_item_matrix, K1=K1, B=B))
    
    logger.info(
        'Матрица подготовлена: плотность = %.4f%%, ненулевых элементов = %d',
        (weighted_matrix.nnz / (weighted_matrix.shape[0] * weighted_matrix.shape[1])) * 100,
        weighted_matrix.nnz
    )
    
    return weighted_matrix


def save_data(path: Path, data: sparse.spmatrix | pd.DataFrame | np.ndarray) -> None:
    """
    Сохранение данных в файл.
    
    Функция определяет тип данных (разреженная матрица, DataFrame или массив NumPy)
    и сохраняет их в соответствующий формат:
    - разреженные матрицы (csr, csc, coo) -> .npz
    - DataFrame и массивы NumPy -> .csv
    
    Параметры
    ----------
    path : Path
        Путь для сохранения файла. Если расширение не соответствует типу данных,
        оно будет автоматически заменено на корректное (.npz или .csv)
    data : scipy.sparse.spmatrix or pd.DataFrame or np.ndarray
        Данные для сохранения. Поддерживаются разреженные матрицы SciPy,
        pandas DataFrame и массивы NumPy
    
    Возвращает
    -------
    None
    
    Исключения
    ------
    OSError
        Если не удалось сохранить файл (ошибка записи, недостаточно прав и т.д.)
    TypeError
        Если тип данных не поддерживается (не sparse matrix, не DataFrame, не ndarray)
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        logger.info('Файл %s уже существует, пропуск сохранения', path)
        return
    
    if sparse.isspmatrix(data):
        if path.suffix != '.npz':
            new_path = path.with_suffix('.npz')
            logger.info('Расширение изменено с %s на .npz для разреженной матрицы', path.suffix)
            path = new_path
        
        try:
            logger.info('Сохранение разреженной матрицы в %s', path)
            sparse.save_npz(path, data)
            logger.info('Разреженная матрица успешно сохранена')
            return
        except OSError as e:
            logger.exception('Ошибка при сохранении разреженной матрицы в %s', path)
            raise OSError(f"Не удалось сохранить разреженную матрицу: {e}") from e
    
    if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        if path.suffix != '.csv':
            new_path = path.with_suffix('.csv')
            logger.info('Расширение изменено с %s на .csv для табличных данных', path.suffix)
            path = new_path
        
        try:
            logger.info('Сохранение табличных данных в %s', path)
            if isinstance(data, np.ndarray):
                logger.debug('Преобразование массива NumPy в DataFrame')
                data = pd.DataFrame(data)
            data.to_csv(path, sep=',', index=False, encoding='utf-8')
            logger.info('Табличные данные успешно сохранены: %d строк, %d столбцов', 
                       len(data), len(data.columns))
            return
        except OSError as e:
            logger.exception('Ошибка при сохранении табличных данных в %s', path)
            raise OSError(f"Не удалось сохранить табличные данные: {e}") from e
    
    raise TypeError(
        f"Неподдерживаемый тип данных: {type(data).__name__}. "
        "Ожидается: scipy.sparse.spmatrix, pandas.DataFrame или numpy.ndarray"
    )


def load_data(path: Path) -> sparse.spmatrix | pd.DataFrame:
    """
    Загрузка данных из файла с автоматическим определением формата.
    
    Функция определяет формат файла по расширению и загружает данные
    в соответствующий тип:
    - .npz -> разреженная матрица SciPy
    - .csv -> pandas DataFrame
    
    Параметры
    ----------
    path : Path
        Путь к файлу для загрузки. Поддерживаются расширения .npz и .csv
    
    Возвращает
    -------
    scipy.sparse.spmatrix or pd.DataFrame
        Загруженные данные:
        - для .npz: разреженная матрица (csr_matrix)
        - для .csv: pandas DataFrame
    
    Исключения
    ------
    FileNotFoundError
        Если файл не существует по указанному пути
    ValueError
        Если расширение файла не поддерживается (.npz или .csv)
    OSError
        Если не удалось прочитать файл (повреждён, недостаточно прав и т.д.)
    """
    
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    logger.info('Загрузка данных из %s', path)
    
    if path.suffix == '.npz':
        try:
            data = sparse.load_npz(path)
            logger.info(
                'Загружена разреженная матрица: форма = %s, ненулевых элементов = %d, плотность = %.4f%%',
                data.shape,
                data.nnz,
                (data.nnz / (data.shape[0] * data.shape[1])) * 100 if data.shape[0] * data.shape[1] > 0 else 0
            )
            return data
        except Exception as e:
            logger.exception('Ошибка при загрузке разреженной матрицы из %s', path)
            raise OSError(f"Не удалось загрузить разреженную матрицу: {e}") from e
    
    if path.suffix == '.csv':
        try:
            data = pd.read_csv(path, encoding='utf-8')
            logger.info(
                'Загружен DataFrame: форма = %s, колонки = %s',
                data.shape,
                list(data.columns)
            )
            return data
        except Exception as e:
            logger.exception('Ошибка при загрузке CSV из %s', path)
            raise OSError(f"Не удалось загрузить CSV файл: {e}") from e
    
    raise ValueError(
        f"Неподдерживаемый формат файла: {path.suffix}. "
        "Ожидается: .npz (разреженная матрица) или .csv (DataFrame)"
    )