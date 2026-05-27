
# Ноутбуки проекта

## Обзор

- `01_eda.ipynb` - разведочный анализ данных MovieLens 32M;
- `02_preprocessing_and_experiments.ipynb` - подготовка данных, эксперименты с моделями, выбор гиперпараметров.

---

### 1. `01_eda.ipynb` — Разведочный анализ данных

**Цель:** Изучить структуру датасета, выявить особенности и проблемы.

**Что делается:**
- Скачивание и распаковка датасета.
- Анализ файлов `ratings.csv`, `movies.csv`, `tags.csv`.
- Выявление пропусков и дубликатов.
- Визуализация распределения фильмов по годам.

**Ключевые выводы:**
- 32 млн оценок, 200948 пользователей, 87585 фильмов.
- 17 пропусков в тегах (заменены на пустые строки).
- У 615 фильмов отсутствует год выпуска.
- Диапазон рейтингов: 0.5 - 5.0.
- Тренд: фильмов становится больше с каждым десятилетием.

**Сохраненные артефакты:**
- `artifacts/notebooks_artifacts/hist_movie_year_plot.png` — гистограмма распределения фильмов по годам

---

### 2. `02_preprocessing_and_experiments.ipynb` — Подготовка данных и эксперименты

**Цель:** Подготовить данные для моделей, провести эксперименты, выбрать лучшие модели.

**Основные этапы:**

#### 2.1. Предобработка признаков
- Извлечение года выпуска из названия фильма.
- Формирование десятилетий (1990s, 2000s и т.д.).
- Очистка тегов (удаление спецсимволов, приведение к нижнему регистру).
- Создание комбинированных текстовых признаков:
  - `genres_decade` — жанры + десятилетие;
  - `genres_decade_tags` — жанры + десятилетие + теги.

#### 2.2. Подготовка выборок
Три варианта выборок для контент-базированных экспериментов:
- Все рейтинги (0.5-5.0).
- Не негативные (3.0-5.0).
- Положительные (4.0-5.0).

#### 2.3. Эксперименты с content-based моделями
Сравнение различных методов векторизации:
- One-hot жанры;
- TF-IDF на жанрах + десятилетиях;
- TF-IDF + SVD (95% дисперсии);
- TF-IDF + SVD (100 компонент).

#### 2.4. Эксперименты с collaborative filtering моделью
- ALS (Alternating Least Squares) из библиотеки `implicit`.
- BM25 взвешивание матрицы user-item.
- Перебор гиперпараметров (factors, regularization, iterations, alpha).

**Сохраненные артефакты:**
- `data/processed/movies_processed.csv` — обработанные признаки фильмов;
- `data/processed/*_to_idx.json` — маппинги ID → индексы;
- `data/processed/cf_user_item_matrix.npz` — взвешенная матрица user-item;
- `data/features/cb_features.csv` — признаки фильмов;
- `artifacts/notebooks_artifacts/baseline_metrics.json`;
- `artifacts/notebooks_artifacts/content_based_metrics.json`;
- `artifacts/notebooks_artifacts/collaborative_filtering_metrics.json`;
- `artifacts/notebooks_artifacts/best_cb_model_config.json`;
- `artifacts/notebooks_artifacts/best_cf_model_config.json`.

**Финальные модели:**

**Content-based**: TF-IDF + TruncatedSVD (n_components=100), обучена на не негативных рейтингах (3.0-5.0).
**Collaborative**: ALS (factors=100, reg=0.1, iter=50, alpha=1.0), BM25 вес, обучена на положительных рейтингах (4.0-5.0).

---

## Запуск ноутбуков

```bash
cd project
# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить dev зависимости
pip install -e ".[dev]"

# Запустить Jupyter
jupyter lab
# или
jupyter notebook
```

**Порядок запуска:**
1. Сначала `01_eda.ipynb`.
2. Затем `02_preprocessing_and_experiments.ipynb`.

**Время выполнения:**
- `01_eda.ipynb`: ~2-3 минуты
- `02_preprocessing_and_experiments.ipynb`: ~1-2 часа (в основном из-за экспериментов)

---

## Перенос кода в `src/`

Логика из ноутбуков, которая используется в сервисе, вынесена в модули.
