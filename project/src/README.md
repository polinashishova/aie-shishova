# Исходный код проекта

В этой папке размещается **основной код проекта**, который используется для:

- подготовки данных;
- обучения моделей;
- инференса (получения предсказаний);
- запуска сервисов (API).

---

## Структура модуля `mrh`

```
src/mrh/
├── __init__.py              # Инициализация пакета
├── config.py                # Работа с переменными окружения
├── data.py                  # Загрузка, подготовка и обработка данных
├── models.py                # Модели: обучение, рекомендации, сохранение/загрузка
├── utils.py                 # Вспомогательные функции (логирование, JSON, словари)
└── api/                     # FastAPI приложение
    ├── __init__.py
    ├── dependencies.py      # Зависимости: загрузка моделей в память, кэширование
    ├── main.py              # Создание FastAPI приложения (lifespan, CORS, роутеры)
    ├── schemas.py           # Pydantic модели для валидации запросов/ответов
    └── endpoints/           # Эндпоинты API
        ├── __init__.py
        ├── health.py        # /health — проверка здоровья сервиса
        └── predict.py       # /predict — получение рекомендаций
```

---

## Модули и их назначение

### 1. `config.py` — Конфигурация

**Назначение:** загрузка переменных окружения и валидация обязательных параметров.

**Что делает:**
- Загружает `.env` файл (если существует)
- Предоставляет константы: `APP_HOST`, `APP_PORT`, `LOG_LEVEL`, `CB_THRESHOLD`, `CF_THRESHOLD`, `CONFIGS_DIR`

**Пример использования:**
```python
from mrh.config import APP_HOST, APP_PORT, LOG_LEVEL
```

---

### 2. `data.py` — Работа с данными

**Назначение:** Загрузка, предобработка и сохранение данных.

**Основные функции:**
| Функция | Описание |
|---------|----------|
| `download_data(url, path)` | Скачивание файла по URL |
| `extract_archive(path_from, directory, expected_path)` | Распаковка ZIP с защитой от path traversal |
| `preprocess_tags(tags_string)` | Очистка тегов (нижний регистр, удаление спецсимволов) |
| `feature_preparation_cb(movies, tags, ratings)` | Подготовка текстовых признаков фильмов |
| `get_ratings_by_threshold(ratings, threshold)` | Фильтрация рейтингов по порогу |
| `data_preparation_cf(ratings, K1, B)` | Создание BM25-взвешенной матрицы user-item |
| `save_data(path, data)` | Сохранение данных (CSV для таблиц, NPZ для матриц) |
| `load_data(path)` | Загрузка данных с автоматическим определением формата |

---

### 3. `models.py` — Модели

**Назначение:** Построение, обучение и использование моделей.

**Основные компоненты:**

#### Контент-базированная модель
| Функция | Описание |
|---------|----------|
| `build_cb_pipeline(parameters)` | Создание пайплайна (TF-IDF + TruncatedSVD) |
| `train_cb_pipeline(pipeline, data, column)` | Обучение пайплайна на текстовых данных |
| `apply_fitted_cb_pipeline(pipeline, data, column)` | Применение обученного пайплайна |
| `recommend_by_movieId(movieIds, movie_features, ...)` | Поиск похожих фильмов через косинусную близость |

#### Коллаборативная модель (ALS)
| Функция | Описание |
|---------|----------|
| `build_als_estimator(parameters)` | Создание ALS модели |
| `train_als_estimator(als_estimator, data)` | Обучение ALS на разреженной матрице |
| `recommend_by_userId(userIds, als_model, ...)` | Получение персонализированных рекомендаций |

#### Сохранение/загрузка
| Функция | Описание |
|---------|----------|
| `save_model(model, path)` | Сохранение модели через joblib |
| `load_model(path)` | Загрузка модели из файла |

---

### 4. `utils.py` — Вспомогательные функции

**Назначение:** Утилиты, используемые во всем проекте.

| Функция | Описание |
|---------|----------|
| `setup_logging(level, log_dir, log_filename)` | Настройка логирования (консоль + файл) |
| `load_json(path)` | Загрузка JSON файла |
| `save_json(data, path)` | Сохранение данных в JSON |
| `inverse_dict(dictionary)` | Инвертирование словаря (ключи <-> значения) |

---

### 5. `api/` — FastAPI приложение

#### `api/dependencies.py` — Зависимости
| Функция | Описание |
|---------|----------|
| `is_models_ready()` | Проверка, загружены ли модели в кэш |
| `get_config_paths()` | Загрузка `paths.json` (кэшируется) |
| `load_cb_model()` | Загрузка CB модели (кэшируется) |
| `load_cf_model()` | Загрузка CF модели (кэшируется) |
| `get_cb_model_dep()` | Dependency для эндпоинтов |
| `get_cf_model_dep()` | Dependency для эндпоинтов |

#### `api/schemas.py` — Pydantic модели
| Модель | Описание |
|--------|----------|
| `PredictRequest` | Валидация запроса: `movieIds`/`userIds`, `k` |
| `PredictResponse` | Структура ответа с рекомендациями |
| `HealthResponse` | Структура ответа `/health` |
| `RecommendationItem` | Отдельная рекомендация (movieId, score) |

#### `api/endpoints/health.py` — Эндпоинт `/health`
- `GET /health` — проверка состояния сервиса и загрузки моделей

#### `api/endpoints/predict.py` — Эндпоинт `/predict`
- `POST /predict` — получение рекомендаций по `movieIds` или `userIds`

#### `api/main.py` — FastAPI приложение
| Функция | Описание |
|---------|----------|
| `lifespan(app)` | Управление жизненным циклом (загрузка/очистка моделей) |
| `create_app()` | Создание и настройка FastAPI приложения |

---

## Как использовать модули

### Импорт в коде

```python
# Импорт данных
from mrh.data import load_data, save_data, preprocess_tags

# Импорт моделей
from mrh.models import build_als_estimator, recommend_by_movieId

# Импорт утилит
from mrh.utils import setup_logging, load_json, save_json, inverse_dict

# Импорт API
from mrh.api.main import create_app
from mrh.api.schemas import PredictRequest, PredictResponse
```

---

## Безопасность и обработка ошибок

В модулях реализованы:

- **Защита от path traversal** — в `extract_archive()`.
- **Типизация** — полные аннотации типов.
- **Валидация входных данных** — проверка типов и граничных значений.
- **Обработка ошибок** — конкретные исключения с понятными сообщениями.
- **Логирование** — информативные сообщения на всех этапах.


---

## Тестирование

Для тестирования модулей используются тесты в `tests/`:

```bash
# Тестирование data.py
pytest tests/test_data.py -v

# Тестирование models.py
pytest tests/test_models.py -v

# Тестирование utils.py
pytest tests/utils.py -v

# Тестирование API
pytest tests/test_api/ -v
```
