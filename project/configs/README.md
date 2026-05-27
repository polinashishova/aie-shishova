# Конфигурационные файлы

## Файлы конфигурации

| Файл | Формат | Назначение |
|------|--------|------------|
| `paths.json` | JSON | Пути к директориям и файлам проекта |
| `models.json` | JSON | Гиперпараметры моделей (SVD, ALS) |
| `.env.example` | ENV | Шаблон переменных окружения |

## Содержимое файлов

**`paths.json`** — конфигурация путей:
```json
{
    "data_dir": "data",
    "data_raw_dir": "data/raw",
    "data_processed_dir": "data/processed",
    "data_features_dir": "data/features",
    "artifacts_dir": "artifacts",
    "models_dir": "models",
    "notebooks_artifacts_dir": "notebooks_artifacts",
    "data_url": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "ml_32m_zip": "ml_32m.zip",
    "ml_32m": "ml-32m",
    "ratings_path": "ratings.csv",
    "movies_path": "movies.csv",
    "tags_path": "tags.csv",
    "cb_pipeline": "cb_pipeline.joblib",
    "als_model": "als_model.joblib",
    "cb_features": "cb_features.csv",
    "cf_user_item_matrix": "cf_user_item_matrix.npz"
}
```

**`models.json`** — гиперпараметры моделей:
```json
{
    "cb": {
        "svd": {
            "n_components": 100,
            "random_state": 42
        }
    },
    "cf": {
        "als": {
            "factors": 100,
            "regularization": 0.1,
            "iterations": 50,
            "alpha": 1.0,
            "random_state": 42
        }
    }
}
```

**`.env.example`** — шаблон переменных окружения:
```bash
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CONFIGS_DIR=./configs/
CB_FEATURE_COLUMN=genres_decade_tags
CB_THRESHOLD=3.0
CF_THRESHOLD=4.0
PYTHONUNBUFFERED=1
OPENBLAS_NUM_THREADS=1
```

## Как использовать

1. **Для запуска проекта:**
```bash
# Скопировать шаблон и заполнить при необходимости
cp configs/.env.example .env
```

2. **Для изменения гиперпараметров:**
   - Отредактировать `configs/models.json`
   - Удалить (если есть) файлы:
     - `artifacts/models/cb_pipeline.joblib`;
     - `artifacts/models/als_model.joblib`;
     - `data/features/cb_features.csv`.
  
```bash
cd project

rm -f artifacts/models/cb_pipeline.joblib
rm -f artifacts/models/als_model.joblib
rm -f data/features/cb_features.csv

# powershell
ri data/features/cb_features.csv 
ri artifacts/models/als_model.joblib
ri data/features/cb_features.csv
```

   - Переобучить модели: `python scripts/train_models.py` (если нет данных, сначала запустить `python scripts/load_and_prepare_data.py`).
 ```bash
python scripts/load_and_prepare_data.py
python scripts/train_models.py
```

3. **Для изменения путей:**
   - Отредактировать `configs/paths.json`


## Загрузка конфигурации в коде

```python
# Пример из mrh/config.py
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env")

APP_HOST = os.environ.get('APP_HOST')
APP_PORT = int(os.environ.get('APP_PORT'))
```

```python
# Пример из mrh/api/dependencies.py
paths = load_json(root_path / CONFIGS_DIR / 'paths.json')
models_config = load_json(root_path / CONFIGS_DIR / 'models.json')
```




## Что можно настроить без изменения кода

| Что настраивается | Файл | Параметр |
|------------------|------|----------|
| Хост и порт сервера | `.env` | `APP_HOST`, `APP_PORT` |
| Уровень логирования | `.env` | `LOG_LEVEL` |
| Пороги рейтингов | `.env` | `CB_THRESHOLD`, `CF_THRESHOLD` |
| Колонка для текстовых признаков | `.env` | `CB_FEATURE_COLUMN` |
| Пути к данным | `paths.json` | все пути |
| Размерность эмбеддингов | `models.json` | `cf.als.factors` |
| Сила регуляризации ALS | `models.json` | `cf.als.regularization` |
| Количество компонент SVD | `models.json` | `cb.svd.n_components` |
| Количество итераций ALS | `models.json` | `cf.als.iterations` |

---

