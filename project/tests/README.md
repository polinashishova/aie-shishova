# Тесты проекта

В этой папке находятся **тесты** для проверки корректности работы кода:

- модульные тесты для функций и классов из `src/`;
- интеграционные тесты для API эндпоинтов;
- тесты с моками для изолированной проверки.

---

## Структура тестов

```
tests/
├── __init__.py
├── test_utils.py               # Unit-тесты для утилит (inverse_dict, JSON, логирование)
├── test_data.py                # Unit-тесты для обработки данных (preprocess_tags, фильтрация)
├── test_models.py              # Unit-тесты для моделей (рекомендации, сохранение/загрузка)
└── test_api/                   # Интеграционные тесты API
    ├── __init__.py
    ├── test_health.py          # Тесты эндпоинта /health
    └── test_predict.py         # Тесты эндпоинта /predict
```

---

## Типы тестов

| Тип | Файлы | Описание | Количество |
|-----|-------|----------|------------|
| **Unit-тесты** | `test_utils.py`, `test_data.py`, `test_models.py` | Проверка изолированных функций | ~30 |
| **Интеграционные тесты** | `test_api/test_health.py`, `test_api/test_predict.py` | Проверка API эндпоинтов | ~10 |
| **Тесты с моками** | `test_models.py`, `test_api/test_predict.py` | Проверка с подменой зависимостей | ~5 |

**Всего тестов:** 55

---

## Что проверяется

### `test_utils.py` — Утилиты

| Класс/Функция | Что проверяет |
|---------------|---------------|
| `TestInverseDict` | Инвертирование словарей, дубликаты, ошибки |
| `TestJsonOperations` | Сохранение и загрузка JSON, обработка ошибок |
| `TestSetupLogging` | Создание директорий для логов, возврат логгера |

### `test_data.py` — Данные

| Класс/Функция | Что проверяет |
|---------------|---------------|
| `TestPreprocessTags` | Очистка тегов (пунктуация, регистр, NaN, None, русские буквы) |
| `TestGetRatingsByThreshold` | Фильтрация рейтингов по порогу, граничные случаи |
| `TestSaveLoadData` | Сохранение/загрузка DataFrame, numpy array, обработка ошибок |

### `test_models.py` — Модели

| Класс/Функция | Что проверяет |
|---------------|---------------|
| `TestRecommendByMovieId` | Рекомендации для 1 и N фильмов, валидация k, ошибки |
| `TestRecommendByUserId` | Рекомендации для пользователей, исключение фильмов |
| `TestSaveLoadModel` | Сохранение/загрузка Pipeline и ALS моделей |
| `TestBuildCBPipeline` | Создание пайплайна с разными параметрами |
| `TestTrainCBPipeline` | Обучение на Series и DataFrame, обработка ошибок |
| `TestBuildALSEstimator` | Создание ALS с разными параметрами |

### `test_api/test_health.py` — Health эндпоинт

| Тест | Что проверяет |
|------|---------------|
| `test_health_endpoint_returns_200` | Статус ответа 200 |
| `test_health_response_format` | Структура JSON ответа |
| `test_root_endpoint` | Корневой эндпоинт `/` |

### `test_api/test_predict.py` — Predict эндпоинт

| Тест | Что проверяет |
|------|---------------|
| `test_predict_without_ids` | Ошибка при отсутствии movieIds и userIds |
| `test_predict_with_both_ids` | Ошибка при указании обоих типов ID |
| `test_predict_with_empty_movieIds` | Ошибка при пустом списке movieIds |
| `test_predict_with_empty_userIds` | Ошибка при пустом списке userIds |
| `test_predict_with_negative_ids` | Ошибка при отрицательных ID |
| `test_predict_with_k_out_of_range` | Ошибка при k вне диапазона (1-100) |
| `test_predict_response_format` | Структура ответа (если модели загружены) |

---

## Запуск тестов

### Все тесты
```bash
cd project
.venv\Scripts\activate      # Windows
# или
source .venv/bin/activate   # Linux/macOS

pytest tests/ -v
```

### С отчетом о покрытии
```bash
pytest tests/ --cov=mrh --cov-report=term
```

### Подробный отчет с пропущенными строками
```bash
pytest tests/ --cov=mrh --cov-report=term-missing
```

### Только unit-тесты
```bash
pytest tests/test_utils.py tests/test_data.py tests/test_models.py -v
```

### Только API тесты
```bash
pytest tests/test_api/ -v
```

### Конкретный класс тестов
```bash
pytest tests/test_utils.py::TestInverseDict -v
```

### Конкретный тест
```bash
pytest tests/test_data.py::TestPreprocessTags::test_normal_string -v
```

---

## Результаты тестов

```
============================= 55 passed, 1 warning in 16.79s =============================

Name                           Stmts   Miss  Cover
-------------------------------------------------------
src\mrh\api\dependencies.py        78     10    87%
src\mrh\api\endpoints\health.py    16      1    94%
src\mrh\api\endpoints\predict.py   42     13    69%
src\mrh\api\main.py                36     17    53%
src\mrh\api\schemas.py             34      0   100%
src\mrh\config.py                  15      0   100%
src\mrh\data.py                   211    137    35%
src\mrh\models.py                 270    133    51%
src\mrh\utils.py                   65     11    83%
-------------------------------------------------------
TOTAL                             767    322    58%
```

---

## Известные предупреждения

При запуске тестов может появляться предупреждение:

```
RuntimeWarning: OpenBLAS is configured to use 12 threads. 
It is highly recommended to disable its internal threadpool...
```

**Это нормально для тестов.** В Docker переменные уже установлены в `OPENBLAS_NUM_THREADS=1`.
