# Артефакты проекта

## Сохранённые модели
| Файл | Размер | Описание |
|------|--------|----------|
| `models/cb_pipeline.joblib` | ~40 MB | TF-IDF + TruncatedSVD (100 компонент) пайплайн для content-based рекомендаций |
| `models/als_model.joblib` | ~100 MB | Обученная ALS модель (100 факторов) |

## Данные для инференса
| Файл | Описание |
|------|----------|
| `data/processed/cb_movieId_to_idx.json` | Маппинг movieId → индекс для CB модели |
| `data/processed/cf_movieId_to_idx.json` | Маппинг movieId → индекс для CF модели |
| `data/processed/cf_userId_to_idx.json` | Маппинг userId → индекс для CF модели |
| `data/processed/cf_user_item_matrix.npz` | Разреженная матрица user-item (BM25 взвешенная) |
| `data/features/cb_features.csv` | Признаки фильмов после SVD (100 компонент) |

## Результаты экспериментов

Лежат в папке `notebooks_artifacts/`.

| Файл | Описание |
|------|----------|
| `notebooks_artifacts/baseline_metrics.json` | Метрики случайной baseline модели |
| `notebooks_artifacts/content_based_metrics.json` | Результаты всех CB экспериментов |
| `notebooks_artifacts/collaborative_filtering_metrics.json` | Результаты ALS гиперпараметров |
| `notebooks_artifacts/best_cb_model_config.json` | Конфигурация лучшей CB модели |
| `notebooks_artifacts/best_cf_model_config.json` | Конфигурация лучшей CF модели |

## Визуализации
| Файл | Описание |
|------|----------|
| `notebooks_artifacts/hist_movie_year_plot.png` | Гистограмма распределения фильмов по годам |

## Логирование

При запуске скриптов логи пишутся в файл `artifacts/logs/logs.log` (название может быть изменено). В репозиторий файл не попадает.

---

## Внешнее хранение

Финальные модели и обработанные данные загружены на **Hugging Face Hub**:

- Репозиторий: `shajeless/movie-recsys-hybrid`
- Ссылка: https://huggingface.co/shajeless/movie-recsys-hybrid

Для загрузки моделей используется скрипт `scripts/download_models.py`.

```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
# загрузка моделей и необходимых данных
python scripts/download_models.py
```

---

## Примечание

Артефакты (особенно модели и `.npz` файлы) не хранятся в Git-репозитории из-за большого размера. 
Для воспроизведения результатов используйте скрипт загрузки или запустите подготовку данных и обучение через `scripts/load_and_prepare_data.py` и `scripts/train_models.py`.

```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
# Загрузка и подготовка данных
python scripts/load_and_prepare_data.py
# Обучение моделей
python scripts/train_models.py
```

Также доступен вариант получения обученных моделей и необходимых для инференса данных через запуск ноутбуков.

```bash
cd project
pip install -e ".[dev]"
jupyter lab notebooks/
```