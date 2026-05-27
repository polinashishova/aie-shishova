"""src/mrh/models.py
Работа с моделями рекомендательной системы: построение моделей (контентно-базированная для user cold-start 
и коллаборативная фильтрация для персонализированных рекомендаций); обучение и применение моделей;
получение с их помощью рекомендаций; сохранение и загрузка моделей.
"""

import logging
from pathlib import Path
from typing import Any, Hashable, Optional

from implicit.cpu.als import AlternatingLeastSquares
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_cb_pipeline(pipeline_parameters: Optional[dict[str, Any]] = None) -> Pipeline:
    """
    Построение пайплайна для контент-базированной модели.
    
    Функция создаёт пайплайн, состоящий из двух этапов:
    1. TfidfVectorizer - преобразование текстовых признаков в TF-IDF матрицу
    2. TruncatedSVD - снижение размерности для уменьшения вычислительной сложности
    
    Параметры
    ----------
    pipeline_parameters : dict, default=None
        Словарь с параметрами для компонентов пайплайна.
        Поддерживаемые ключи:
        - 'tfidf': параметры для TfidfVectorizer
        - 'svd': параметры для TruncatedSVD
        Если параметры не указаны, используются значения по умолчанию.
    
    Возвращает
    -------
    Pipeline
        Scikit-learn пайплайн с настроенными компонентами:
        - 'tfidf': TfidfVectorizer
        - 'svd': TruncatedSVD
    
    Исключения
    ------
    TypeError
        Если pipeline_parameters не является dict или не является None
    """
    
    if pipeline_parameters is not None and not isinstance(pipeline_parameters, dict):
        raise TypeError(
            f"pipeline_parameters должен быть dict или None, получен {type(pipeline_parameters).__name__}"
        )
    
    if pipeline_parameters is None:
        pipeline_parameters = {}
        logger.debug('pipeline_parameters не указаны, используются параметры по умолчанию')
    
    tfidf_params = pipeline_parameters.get('tfidf', {})
    svd_params = pipeline_parameters.get('svd', {})
    
    logger.info(
        'Создание CB пайплайна с переданными параметрами: TfidfVectorizer(%s), TruncatedSVD(%s)',
        tfidf_params,
        svd_params
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('svd', TruncatedSVD(**svd_params))
    ])
    
    logger.info('Пайплайн успешно создан')
    
    return pipeline


def train_cb_pipeline(
    pipeline: Pipeline, 
    data: pd.Series | pd.DataFrame, 
    column: Optional[Hashable] = None
) -> Pipeline:
    """
    Обучение контент-базированного пайплайна на текстовых данных.
    
    Функция принимает обученный пайплайн (TfidfVectorizer + TruncatedSVD)
    и обучает его на текстовых данных. Поддерживает входные данные в виде
    pandas Series или DataFrame с указанием колонки.
    
    Параметры
    ----------
    pipeline : Pipeline
        Scikit-learn пайплайн с компонентами 'tfidf' и 'svd'
    data : pd.Series or pd.DataFrame
        Входные данные:
        - Если pd.Series: используется напрямую как текстовые данные
        - Если pd.DataFrame: используется колонка, указанная в параметре column
    column : Hashable, default=None
        Название колонки в DataFrame, содержащей текстовые данные.
        Обязателен, если data является DataFrame.
    
    Возвращает
    -------
    Pipeline
        Обученный пайплайн (fit-модель)
    
    Исключения
    ------
    TypeError
        Если pipeline не является экземпляром Pipeline
        Если data не является pd.Series или pd.DataFrame
    ValueError
        Если data является DataFrame и column не указан
        Если column не найден в DataFrame
        Если после преобразования данные пусты
    """
    
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"pipeline должен быть sklearn.pipeline.Pipeline, получен {type(pipeline).__name__}"
        )
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"data должен быть pd.Series или pd.DataFrame, получен {type(data).__name__}"
        )
    
    if isinstance(data, pd.Series):
        train_data = data.values.tolist()
        logger.info("Обучение на Series с %d текстами", len(train_data))
        
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError(
                "Для DataFrame необходимо указать параметр column с названием колонки"
            )
        
        if column not in data.columns:
            raise ValueError(f"Колонка '{column}' не найдена в DataFrame")
        
        train_data = data[column].values.tolist()
        logger.info(
            "Обучение на колонке '%s' DataFrame: %d текстов из %d строк",
            column, len(train_data), len(data)
        )
    
    if len(train_data) == 0:
        raise ValueError("Нет данных для обучения")
    
    try:
        logger.info("Начало обучения пайплайна")
        trained_pipeline = pipeline.fit(train_data)
        logger.info("Пайплайн успешно обучен")
        return trained_pipeline
    except Exception as e:
        logger.exception("Ошибка при обучении пайплайна")
        raise RuntimeError(f"Не удалось обучить пайплайн: {e}") from e


def apply_fitted_cb_pipeline(
    pipeline: Pipeline, 
    data: pd.Series | pd.DataFrame, 
    column: Optional[Hashable] = None
) -> np.ndarray:
    """
    Применение обученного контент-базированного пайплайна к данным.
    
    Функция применяет обученный пайплайн (TfidfVectorizer + TruncatedSVD)
    для преобразования текстовых данных в признаковое пространство.
    
    Параметры
    ----------
    pipeline : Pipeline
        Обученный scikit-learn пайплайн с компонентами 'tfidf' и 'svd'
    data : pd.Series or pd.DataFrame
        Входные данные:
        - Если pd.Series: используется напрямую как текстовые данные
        - Если pd.DataFrame: используется колонка, указанная в параметре column
    column : Optional[Hashable], default=None
        Название колонки в DataFrame, содержащей текстовые данные.
        Обязателен, если data является DataFrame.
    
    Возвращает
    -------
    np.ndarray
        Матрица признаков размерности (n_samples, n_components) после применения
        TF-IDF и SVD преобразований.
    
    Исключения
    ------
    TypeError
        Если pipeline не является экземпляром Pipeline
        Если data не является pd.Series или pd.DataFrame
    ValueError
        Если data является DataFrame и column не указан
        Если column не найден в DataFrame
        Если после преобразования данные пусты
    RuntimeError
        Если произошла ошибка при применении пайплайна
    """
    
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"pipeline должен быть sklearn.pipeline.Pipeline, получен {type(pipeline).__name__}"
        )
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"data должен быть pd.Series или pd.DataFrame, получен {type(data).__name__}"
        )
    
    if isinstance(data, pd.Series):
        transform_data = data.values.tolist()
        logger.info("Применение пайплайна к Series с %d текстами", len(transform_data))
        
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError(
                "Для DataFrame необходимо указать параметр column с названием колонки"
            )
        
        if column not in data.columns:
            raise ValueError(f"Колонка '{column}' не найдена в DataFrame. Доступные колонки: {list(data.columns)}")
        
        transform_data = data[column].values.tolist()
        logger.info(
            "Применение пайплайна к колонке '%s' DataFrame: %d текстов из %d строк",
            column, len(transform_data), len(data)
        )
    
    if len(transform_data) == 0:
        raise ValueError("Нет данных для преобразования")
    
    try:
        logger.info("Применение обученного пайплайна к данным")
        transformed_features = pipeline.transform(transform_data)
        logger.info(
            "Преобразование завершено: форма матрицы признаков = %s",
            transformed_features.shape
        )
        return transformed_features
    except Exception as e:
        logger.exception("Ошибка при применении пайплайна")
        raise RuntimeError(f"Не удалось применить пайплайн: {e}") from e


def build_als_estimator(parameters: Optional[dict[str, Any]] = None) -> AlternatingLeastSquares:
    """
    Создание ALS (Alternating Least Squares) Estimator для коллаборативной фильтрации.
    
    Функция создаёт и настраивает модель AlternatingLeastSquares из библиотеки implicit.
    
    Параметры
    ----------
    parameters : Optional[dict], default=None
        Словарь с параметрами для модели AlternatingLeastSquares.
        Если None, используются параметры по умолчанию.
        
        Основные параметры ALS:
        - factors : int, default=100
            Количество латентных факторов (размерность эмбеддингов)
        - regularization : float, default=0.01
            Коэффициент регуляризации
        - iterations : int, default=15
            Количество итераций обучения
        - alpha : float, default=1.0
            Параметр уверенности (confidence) для неявных данных  
    Возвращает
    -------
    AlternatingLeastSquares
        Настроенная модель ALS для коллаборативной фильтрации
    
    Исключения
    ------
    TypeError
        Если parameters не является dict или None
    """
    
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError(
            f"parameters должен быть dict или None, получен {type(parameters).__name__}"
        )
    
    if parameters is None:
        parameters = {}
        logger.debug("Используются параметры ALS по умолчанию")
    else:
    
        logger.info(
            "Создание ALS модели с параметрами: %s",
            {k: v for k, v in parameters.items() if k != 'random_state'}
        )
    
    try:
        model = AlternatingLeastSquares(**parameters)
        logger.info("ALS модель успешно создана")
        return model
    except Exception as e:
        logger.exception("Ошибка при создании ALS модели")
        raise RuntimeError(f"Не удалось создать ALS модель: {e}") from e


def train_als_estimator(
    als_estimator: AlternatingLeastSquares, 
    data: sparse.csr_matrix
) -> AlternatingLeastSquares:
    """
    Обучение ALS (Alternating Least Squares) модели на разреженной матрице.
    
    Параметры
    ----------
    als_estimator : AlternatingLeastSquares
        Необученная или частично обученная модель ALS
    data : sparse.csr_matrix
        Разреженная матрица пользователь-фильм в формате CSR.
        Размерность: (n_users, n_items)
    
    Возвращает
    -------
    AlternatingLeastSquares
        Обученная модель ALS
    
    Исключения
    ------
    TypeError
        Если als_estimator не является AlternatingLeastSquares
        Если data не является scipy.sparse.csr_matrix
    ValueError
        Если матрица data пуста или имеет неправильную размерность
    RuntimeError
        Если произошла ошибка в процессе обучения
    """
    
    if not isinstance(als_estimator, AlternatingLeastSquares):
        raise TypeError(
            f"als_estimator должен быть AlternatingLeastSquares, получен {type(als_estimator).__name__}"
        )
    
    if not isinstance(data, sparse.csr_matrix):
        raise TypeError(
            f"data должен быть scipy.sparse.csr_matrix, получен {type(data).__name__}"
        )
    
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"Матрица data имеет неверную размерность: {data.shape}. "
            "Ожидается (n_users, n_items) с n_users > 0 и n_items > 0"
        )
    
    if data.nnz == 0:
        raise ValueError(
            f"Матрица data не содержит ненулевых элементов (nnz = 0). "
            "Невозможно обучить ALS модель на пустой матрице"
        )
    
    try:
        logger.info(
            "Начало обучения ALS модели. Размер матрицы: %d пользователей, %d фильмов, %d взаимодействий",
            data.shape[0], data.shape[1], data.nnz
        )
        als_estimator.fit(data)
        logger.info("Модель ALS успешно обучена")
        return als_estimator
    except Exception as e:
        logger.exception("Ошибка при обучении ALS модели")
        raise RuntimeError(f"Не удалось обучить ALS модель: {e}") from e


def recommend_by_movieId(
    movieIds: int | list[int], 
    movie_features: np.ndarray, 
    movieId_to_idx: dict[int, int], 
    idx_to_movieId: dict[int, int], 
    k: int = 10
) -> tuple[tuple[list[int], list[float]], ...]:
    """
    Получение рекомендаций похожих фильмов на основе movieId.
    
    Функция вычисляет косинусную похожесть между запрашиваемыми фильмами
    и всеми фильмами в матрице признаков, возвращая top-k наиболее похожих.
    
    Параметры
    ----------
    movieIds : int or list[int]
        ID фильма или список ID фильмов, для которых ищем похожие
    movie_features : np.ndarray
        Матрица признаков фильмов размерности (n_movies, n_features)
    movieId_to_idx : dict[int, int]
        Словарь отображения movieId -> индекс в матрице
    idx_to_movieId : dict[int, int]
        Словарь отображения индекс -> movieId
    k : int, default=10
        Количество возвращаемых рекомендаций для каждого фильма
    
    Возвращает
    -------
    tuple[tuple[tuple[int, float], ...], ...]
        Вложенный кортеж рекомендаций. Для каждого запрашиваемого фильма
        возвращается кортеж пар (movieId, similarity_score).
        
        Пример: (( (101, 0.95), (102, 0.89) ), ( (201, 0.92), (202, 0.85) ))
    
    Исключения
    ------
    TypeError
        Если movieIds не является int или list[int]
        Если movie_features не является np.ndarray
        Если k не является int
    ValueError
        Если k <= 0
        Если movie_features пуст
    KeyError
        Если movieId не найден в movieId_to_idx
        Если индекс не найден в idx_to_movieId
    """
    
    if not isinstance(movieIds, (int, list)):
        raise TypeError(
            f"movieIds должен быть int или list[int], получен {type(movieIds).__name__}"
        )
    
    if not isinstance(movie_features, np.ndarray):
        raise TypeError(
            f"movie_features должен быть np.ndarray, получен {type(movie_features).__name__}"
        )
    
    if not isinstance(k, int):
        raise TypeError(f"k должен быть int, получен {type(k).__name__}")
    
    if k <= 0:
        raise ValueError(f"k должен быть больше 0, получен {k}")
    
    if len(movie_features) == 0:
        raise ValueError("Матрица movie_features пуста")
    
    if isinstance(movieIds, int):
        movieIds = [movieIds]

    indices = []
    for movieId in movieIds:
        if movieId not in movieId_to_idx:
            logger.error("Фильм %d не найден в movieId_to_idx", movieId)
            raise KeyError(f"Фильм {movieId} не найден")
        indices.append(movieId_to_idx[movieId])
    
    for idx in indices:
        if idx in idx_to_movieId:
            if idx_to_movieId[idx] in movieIds:
                continue
            logger.error('Фильм %d не соответствует запрашиваемым', idx_to_movieId[idx])
            raise KeyError(f'Фильм {idx_to_movieId[idx]} не соответствует запрашиваемым')
        else:
            logger.error('Индекс %d не найден', idx)
            raise KeyError(f'Индекс {idx} не найден')
    
    logger.info("Поиск рекомендаций для %d фильмов, k=%d, всего фильмов: %d",
                len(indices), k, len(movie_features))
    
    query_movies = movie_features[indices]
    
    similarities = cosine_similarity(query_movies, movie_features)
    
    n_movies = len(movie_features)
    top_k_indices = []
    
    for i in range(len(indices)):
        query_idx = indices[i]
        sim_row = similarities[i]
        
        sim_row[query_idx] = -1
    
        if n_movies <= k:
            top_idx = np.argsort(sim_row)[::-1][:n_movies-1]
        else:
            top_idx = np.argpartition(sim_row, -k)[-k:]
            top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]
        
        top_k_indices.append(top_idx)
    
    recommendations = []
    for i, top_idx in enumerate(top_k_indices):
        rec_movieIds = [idx_to_movieId[idx] for idx in top_idx]
        rec_scores = similarities[i][top_idx].tolist()
        recommendations.append((rec_movieIds, rec_scores))
    
    logger.info(
        "Рекомендации получены: %d запросов, по %d рекомендаций",
        len(recommendations),
        len(recommendations[0][0]) if recommendations else 0
    )
    
    return tuple(recommendations)


def recommend_by_userId(
    userIds: int | list[int], 
    als_model: AlternatingLeastSquares, 
    user_item_matrix: sparse.csr_matrix,
    movieId_to_idx: dict[int, int], 
    userId_to_idx: dict[int, int], 
    idx_to_movieId: dict[int, int], 
    idx_to_userId: dict[int, int], 
    k: int = 10,
    neg_movieIds: Optional[int | list[int] | dict[int, int | list[int]]] = None
) -> tuple[tuple[list[int], list[float]], ...]:
    """
    Получение рекомендаций для пользователя(ей) на основе обученной ALS модели.
    
    Функция возвращает список рекомендуемых фильмов для каждого указанного пользователя,
    с возможностью исключения негативных фильмов (которые не нужно рекомендовать).
    
    Параметры
    ----------
    userIds : int or list[int]
        ID пользователя или список ID пользователей, для которых нужны рекомендации
    als_model : AlternatingLeastSquares
        Обученная ALS модель из библиотеки implicit
    user_item_matrix : sparse.csr_matrix
        Разреженная матрица пользователь-фильм (пользователи × фильмы)
    movieId_to_idx : dict[int, int]
        Словарь отображения movieId -> индекс в матрице
    userId_to_idx : dict[int, int]
        Словарь отображения userId -> индекс в матрице
    idx_to_movieId : dict[int, int]
        Словарь отображения индекс -> movieId
    idx_to_userId : dict[int, int]
        Словарь отображения индекс -> userId
    k : int, default=10
        Количество возвращаемых рекомендаций для каждого пользователя
    neg_movieIds : Optional[int | list[int], dict[int, int | list[int]]], default=None
        Фильмы, которые нужно исключить из рекомендаций.
        Может быть:
        - int: один фильм для первого пользователя
        - list[int]: список фильмов для первого пользователя
        - dict[int, int | list[int]]: словарь {userId: фильм(ы)} для разных пользователей
    
    Возвращает
    -------
    tuple[tuple[list[int], list[float]], ...]
        Кортеж результатов для каждого пользователя. Каждый результат - кортеж из двух элементов:
        - list[int]: список movieId рекомендуемых фильмов
        - list[float]: список оценок 
    
    Исключения
    ------
    TypeError
        Если входные параметры имеют неверный тип
    KeyError
        Если userId не найден в userId_to_idx
        Если movieId из neg_movieIds не найден в movieId_to_idx
        Если индекс не найден в idx_to_movieId или idx_to_userId
    ValueError
        Если k <= 0
    """

    if not isinstance(userIds, (int, list)):
        raise TypeError(
            f"userIds должен быть int или List[int], получен {type(userIds).__name__}"
        )
    
    if not isinstance(als_model, AlternatingLeastSquares):
        raise TypeError(
            f"als_model должен быть AlternatingLeastSquares, получен {type(als_model).__name__}"
        )
    
    if not isinstance(user_item_matrix, sparse.csr_matrix):
        raise TypeError(
            f"user_item_matrix должен быть scipy.sparse.csr_matrix, получен {type(user_item_matrix).__name__}"
        )
    
    if not isinstance(k, int):
        raise TypeError(f"k должен быть int, получен {type(k).__name__}")
    
    if k <= 0:
        raise ValueError(f"k должен быть больше 0, получен {k}")
    
    if isinstance(userIds, int):
        userIds = [userIds]
    
    if neg_movieIds is None:
        neg_movieIds = {}
    elif isinstance(neg_movieIds, int):
        neg_movieIds = {userIds[0]: [neg_movieIds]}
    elif isinstance(neg_movieIds, list):
        neg_movieIds = {userIds[0]: neg_movieIds}
    elif not isinstance(neg_movieIds, dict):
        raise TypeError(
            f"neg_movieIds должен быть int, list, dict или None, получен {type(neg_movieIds).__name__}"
        )
    
    normalized_neg_movieIds = {}
    for uid, movies in neg_movieIds.items():
        if isinstance(movies, int):
            normalized_neg_movieIds[uid] = [movies]
        elif isinstance(movies, list):
            normalized_neg_movieIds[uid] = movies
        else:
            raise TypeError(
                f"Значения в neg_movieIds должны быть int или list, получен {type(movies).__name__}"
            )
    neg_movieIds = normalized_neg_movieIds
    
    logger.info(
        "Получение рекомендаций для %d пользователей, k=%d",
        len(userIds), k
    )

    user_indices = []
    for userId in userIds:
        if userId not in userId_to_idx:
            logger.error("Пользователь %d не найден в userId_to_idx", userId)
            raise KeyError(f"Пользователь {userId} не найден")
        user_indices.append(userId_to_idx[userId])
    
    for idx in user_indices:
        if idx not in idx_to_userId:
            logger.error("Индекс %d не найден в idx_to_userId", idx)
            raise KeyError(f"Индекс {idx} не найден")
        if idx_to_userId[idx] not in userIds:
            logger.warning(
                "Пользователь %d (индекс %d) не соответствует запрашиваемым %s",
                idx_to_userId[idx], idx, userIds
            )
    
    user_filters = {}
    for userId in userIds:
        if userId in neg_movieIds:
            movie_indices = []
            for movieId in neg_movieIds[userId]:
                if movieId not in movieId_to_idx:
                    logger.error("Фильм %d не найден в movieId_to_idx", movieId)
                    raise KeyError(f"Фильм {movieId} не найден")
                movie_idx = movieId_to_idx[movieId]
                
                if movie_idx not in idx_to_movieId:
                    logger.error("Индекс %d не найден в idx_to_movieId", movie_idx)
                    raise KeyError(f"Индекс {movie_idx} не найден")
                if idx_to_movieId[movie_idx] != movieId:
                    logger.warning(
                        "Несоответствие: индекс %d -> %d, ожидался %d",
                        movie_idx, idx_to_movieId[movie_idx], movieId
                    )
                
                movie_indices.append(movie_idx)
            
            user_filters[userId] = movie_indices
            logger.debug(
                "Для пользователя %d будут исключены %d фильмов",
                userId, len(movie_indices)
            )
    
    recommendations = []
    for i, user_idx in enumerate(user_indices):
        userId = userIds[i]
        filter_list = user_filters.get(userId, None)
        
        logger.debug("Запрос рекомендаций для пользователя %d (индекс %d)",
                     userId, user_idx)
        
        try:
            ids, scores = als_model.recommend(
                user_idx, 
                user_item_matrix[user_idx], 
                N=k,
                filter_items=filter_list,
                filter_already_liked_items=True
            )
            
            rel_movieIds = [idx_to_movieId[idx] for idx in ids]
            recommendations.append((rel_movieIds, scores.tolist()))
            
            logger.debug(
                "Для пользователя %d получено %d рекомендаций, средняя оценка: %.4f",
                userId, len(rel_movieIds), scores.mean()
            )
            
        except Exception as e:
            logger.exception("Ошибка при получении рекомендаций для пользователя %d", userId)
            raise RuntimeError(f"Не удалось получить рекомендации для пользователя {userId}: {e}") from e
    
    logger.info("Рекомендации успешно получены для %d пользователей", len(recommendations))
    
    return tuple(recommendations)


def save_model(model: Pipeline | AlternatingLeastSquares, path: Path) -> None:
    """
    Сохранение модели в файл через joblib.
    
    Параметры
    ----------
    model : Pipeline or AlternatingLeastSquares
        Обученная модель для сохранения
    path : Path
        Путь для сохранения. Расширение будет автоматически приведено к .joblib
    
    Исключения
    ----------
    TypeError, OSError, RuntimeError
    """
    
    if not isinstance(model, (Pipeline, AlternatingLeastSquares)):
        raise TypeError(
            f"model должен быть Pipeline или AlternatingLeastSquares, "
            f"получен {type(model).__name__}"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception("Не удалось создать директорию %s", path.parent)
        raise OSError(f"Не удалось создать директорию: {e}") from e
    
    if path.exists():
        logger.info("Файл %s уже существует, пропуск сохранения", path)
        return
    
    if path.suffix != '.joblib':
        logger.info("Расширение изменено с %s на .joblib", path.suffix)
        path = path.with_suffix('.joblib')
    
    logger.info("Сохранение модели %s в %s", type(model).__name__, path)
    
    try:
        joblib.dump(model, path)
        logger.info("Модель успешно сохранена в %s", path)
    except Exception as e:
        logger.exception("Ошибка при сохранении модели в %s", path)
        raise RuntimeError(f"Не удалось сохранить модель: {e}") from e


def load_model(path: Path) -> Pipeline | AlternatingLeastSquares:
    """
    Загрузка модели из файла через joblib.
    
    Параметры
    ----------
    path : Path
        Путь к файлу модели (ожидается расширение .joblib)
    
    Возвращает
    -------
    Pipeline or AlternatingLeastSquares
    
    Исключения
    ----------
    FileNotFoundError, TypeError, RuntimeError
    """
    
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")

    if path.suffix != '.joblib':
        logger.warning("Ожидается расширение .joblib, но получен %s. Пробуем загрузить...", path.suffix)
    
    logger.info("Загрузка модели из %s", path)
    
    try:
        model = joblib.load(path)
        
        if not isinstance(model, (Pipeline, AlternatingLeastSquares)):
            raise TypeError(
                f"Загруженный объект имеет тип {type(model).__name__}, "
                "ожидался Pipeline или AlternatingLeastSquares"
            )
        
        if isinstance(model, AlternatingLeastSquares):
            if model.user_factors is None or model.item_factors is None:
                raise RuntimeError(
                    "Модель ALS загружена, но user_factors или item_factors = None. "
                    "Требуется переобучение или корректное сохранение через joblib."
                )
            logger.info(
                "ALS модель загружена: user_factors=%s, item_factors=%s",
                model.user_factors.shape,
                model.item_factors.shape
            )
        
        logger.info("Модель успешно загружена: %s", type(model).__name__)
        return model
        
    except (FileNotFoundError, TypeError, RuntimeError):
        raise
    except Exception as e:
        logger.exception("Ошибка при загрузке модели из %s", path)
        raise RuntimeError(f"Не удалось загрузить модель: {e}") from e