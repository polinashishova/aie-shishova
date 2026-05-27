# Скрипты проекта

В этой папке находятся скрипты для автоматизации основных задач проекта: загрузка данных, обучение моделей, скачивание артефактов и запуск API сервиса.

---

## Обзор скриптов

- `load_and_prepare_data.py` - загрузка и подготовка данных MovieLens 32M;
- `train_models.py` - обучение и сохранение моделей;
- `download_models.py` - скачивание готовых моделей с Hugging Face Hub;
- `run_api.py` - запуск FastAPI приложения.

---


## Установка зависимостей

```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Для запуска API (минимально)
pip install -e ".[api]"

# Для полного цикла (обучение + API)
pip install -e ".[dev,api]"
```

## Типовые сценарии

### Сценарий 1: Только запуск сервиса (минимальный)

```bash
# 1. Скачать готовые модели и данные
python scripts/download_models.py

# 2. Запустить API
python scripts/run_api.py
```

#### Сценарий 2: Полный цикл (обучение + запуск)

```bash
# 1. Загрузить и подготовить данные
python scripts/load_and_prepare_data.py

# 2. Обучить модели
python scripts/train_models.py

# 3. Запустить API
python scripts/run_api.py
```

#### Сценарий 3: Только эксперименты (ноутбуки)

```bash
# Установить dev зависимости
pip install -e ".[dev]"

# Запустить Jupyter
jupyter lab notebooks/
```

---

## Детальное описание скриптов

### 1. `load_and_prepare_data.py`

**Назначение:** Загрузка датасета MovieLens 32M и подготовка данных для обучения.

**Что делает:**
1. Скачивает `ml-32m.zip` с официального сайта.
2. Распаковывает архив.
3. Фильтрует рейтинги по порогам (CB_THRESHOLD, CF_THRESHOLD из `.env`).
4. Подготавливает текстовые признаки фильмов (жанры, год, десятилетие, теги).
5. Создает маппинги ID → индексы.
6. Сохраняет обработанные данные в `data/processed/`.

**Использование:**

```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/load_and_prepare_data.py
```

**Выходные файлы:**
- `data/processed/movies_processed.csv` — признаки фильмов;
- `data/processed/*_to_idx.json` — маппинги ID;
- `data/processed/cf_user_item_matrix.npz` — матрица user-item.

**Конфигурация через `.env`:**
```bash
CB_THRESHOLD=3.0    # порог для content-based модели
CF_THRESHOLD=4.0    # порог для collaborative filtering
```
**Примечание:** скрипт пропускает уже существующие файлы.

---

### 2. `train_models.py`

**Назначение:** обучение и сохранение контент-базированной и коллаборативной моделей.

**Что делает:**
1. Загружает обработанные данные.
2. Обучает CB пайплайн (TF-IDF + TruncatedSVD).
3. Обучает ALS модель (Alternating Least Squares).
4. Сохраняет модели в `artifacts/models/`.

**Использование:**
```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/train_models.py
```

**Выходные файлы:**
- `artifacts/models/cb_pipeline.joblib` — обученный CB пайплайн;
- `artifacts/models/als_model.joblib` — обученная ALS модель;
- `data/features/cb_features.csv` — признаки фильмов.

**Конфигурация моделей:** `configs/models.json`
```json
{
    "cb": {
        "svd": { "n_components": 100, "random_state": 42 }
    },
    "cf": {
        "als": { "factors": 100, "regularization": 0.1, "iterations": 50, "alpha": 1.0, "random_state": 42 }
    }
}
```
**Примечание:** скрипт пропускает уже существующие файлы.

---

### 3. `download_models.py`

**Назначение:** скачивание готовых моделей и обработанных данных с Hugging Face Hub.

**Что делает:**
1. Проверяет наличие файлов локально
2. При отсутствии скачивает из репозитория `shajeless/movie-recsys-hybrid`
3. Раскладывает файлы по соответствующим директориям

**Использование:**
```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/download_models.py
```

**Что скачивается:**
| Откуда | Куда |
|--------|------|
| `models/cb_pipeline.joblib` | `artifacts/models/` |
| `models/als_model.joblib` | `artifacts/models/` |
| `data/processed/*` | `data/processed/` |
| `data/features/*` | `data/features/` |

**Примечание:** скрипт пропускает уже существующие файлы.

---

### 4. `run_api.py`

**Назначение:** запуск FastAPI сервера с рекомендательным сервисом.

**Что делает:**
1. Загружает модели в память (при старте)
2. Запускает Uvicorn сервер
3. Обрабатывает запросы к эндпоинтам `/predict`, `/health`, `/docs`

**Использование:**
```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Базовый запуск
python scripts/run_api.py

# С параметрами
python scripts/run_api.py --host 0.0.0.0 --port 8000 --workers 2 --reload
```

**Параметры командной строки:**
| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--host` | `0.0.0.0` | Хост для сервера |
| `--port` | `8000` | Порт для сервера |
| `--workers` | `1` | Количество воркеров |
| `--reload` | `False` | Автоперезагрузка при изменении кода (только для разработки) |

**Проверка работы:** см. README.md paзделы 4.3.1. Проверка работоспособности через curl и 4.3.2. Проверка работоспособности через Swagger UI.


---

## Переменные окружения

Скрипты используют переменные из `.env` файла:

```bash
# .env
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CONFIGS_DIR=./configs/

# Параметры моделей
CB_FEATURE_COLUMN=genres_decade_tags
CB_THRESHOLD=3.0
CF_THRESHOLD=4.0

# BLAS оптимизации
PYTHONUNBUFFERED=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```


---

## Примечания

- Скрипты предполагают, что запуск происходит из **корневой директории проекта**.
- Для GPU обучение ALS не требуется (CPU версия из `implicit.cpu`).
```
