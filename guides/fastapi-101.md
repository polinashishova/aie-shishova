# fastapi-101 – минимальный HTTP-сервис для модели

## 1. Зачем нам FastAPI

Во многих реальных задачах ML-система – это не только код обучения и скрипт `predict`, но и **онлайн-сервис**, к которому обращаются другие системы (UI, backend, другие сервисы).

На курсе мы используем **FastAPI**, чтобы:

- сделать минимальный HTTP-сервис для модели;
- научиться описывать **контракты** (формат запросов и ответов);
- научиться разделять:
  - ядро логики (загрузка модели, предсказания),
  - и **обёртку в виде API**.

> Идея: CLI на Typer – это интерфейс для человека/скриптов в терминале, а FastAPI – интерфейс для других сервисов по HTTP.

---

## 2. Что нужно уметь до FastAPI

Перед тем как поднимать сервис на FastAPI, предполагается, что ты:

- работаешь в терминале (см. `unix-101.md`);
- понимаешь структуру проекта и модулей (см. `python-101.md`);
- умеешь запускать код через `uv run` и работать с зависимостями (см. `uv-101.md`);
- представляешь свою логику обучения/предсказаний (см. `cli-101-typer.md` для CLI).

FastAPI – это всего лишь ещё один «тонкий слой» поверх уже написанного кода.

---

## 3. Установка FastAPI и uvicorn

Если в проекте ещё нет FastAPI и uvicorn, добавляем:

```bash
uv add fastapi "uvicorn[standard]"
```

После этого их можно импортировать в коде и запускать через `uv run uvicorn ...`.

---

## 4. Минимальный сервис: только health-check

Начнём с самого простого: сервис с одним эндпоинтом `/health`, показывающим, что приложение живо.

Предположим, у нас есть структура:

```text
project/
  src/
    project/
      __init__.py
      api.py       # здесь будет FastAPI-приложение
```

Содержимое `project/src/project/api.py`:

```python
# project/src/project/api.py
from fastapi import FastAPI

app = FastAPI(title="Demo ML Service", version="0.1.0")


@app.get("/health")
def health():
    """
    Простой health-check сервиса.
    """
    return {"status": "ok"}
```

### 4.1. Запуск сервиса

Из корня репозитория:

```bash
uv run uvicorn project.api:app --reload --port 8000
```

Где:

- `project.api:app` – это `project/src/project/api.py` и объект `app` внутри;
- `--reload` – режим авто-перезапуска при изменении кода (для разработки);
- `--port 8000` – порт, на котором слушает сервис.

После запуска:

- `http://127.0.0.1:8000/health` → `{"status": "ok"}`
- `http://127.0.0.1:8000/docs` → авто-сгенерированная Swagger-документация
- `http://127.0.0.1:8000/openapi.json` → JSON-описание API

---

## 5. Добавляем модель: /predict с Pydantic-моделями

Обычно нам нужен не только `/health`, но и эндпоинт `/predict`, который принимает данные, прогоняет через модель и возвращает результат.

### 5.1. Pydantic-модели запроса и ответа

Опишем формат входных данных и ответа как Pydantic-классы:

```python
# project/src/project/api.py
from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """
    Формат тела запроса на предсказание.
    Пример: {"feature1": 1.23, "feature2": 4.56}
    """
    feature1: float
    feature2: float


class PredictResponse(BaseModel):
    """
    Формат ответа сервиса.
    """
    prediction: float
    model_version: str | None = None
```

### 5.2. Заготовка для модели

Логику загрузки модели и предсказания лучше держать **не** в самом API-файле:

```python
# project/src/project/core.py
# Здесь может быть логика загрузки/кеширования модели, препроцессинга и т.д.

from functools import lru_cache


class DummyModel:
    def __init__(self, version: str = "0.1.0"):
        self.version = version

    def predict(self, feature1: float, feature2: float) -> float:
        # Заглушка: в реальном коде будет модель
        return feature1 + feature2


@lru_cache(maxsize=1)
def load_model() -> DummyModel:
    # В реальном коде можно грузить модель с диска, из артефакта и т.п.
    return DummyModel(version="0.1.0")
```

### 5.3. Эндпоинт `/predict`

Теперь связываем всё в `api.py`:

```python
# project/src/project/api.py
from fastapi import FastAPI
from pydantic import BaseModel

from .core import load_model


app = FastAPI(title="Demo ML Service", version="0.1.0")


class PredictRequest(BaseModel):
    feature1: float
    feature2: float


class PredictResponse(BaseModel):
    prediction: float
    model_version: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Сделать предсказание по признакам feature1, feature2.
    """
    model = load_model()
    y = model.predict(feature1=request.feature1, feature2=request.feature2)
    return PredictResponse(prediction=y, model_version=model.version)
```

Запуск (как и раньше):

```bash
uv run uvicorn project.api:app --reload --port 8000
```

Пример запроса:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.5, "feature2": 2.0}'
```

Пример ответа:

```json
{
  "prediction": 3.5,
  "model_version": "0.1.0"
}
```

---

## 6. Рекомендованная структура проекта с API

Типичный вариант для курса:

```text
project/
  src/
    project/
      __init__.py
      core.py        # логика модели / обработки данных
      cli.py         # Typer CLI (см. cli-101-typer)
      api.py         # FastAPI-приложение
  configs/
    train.yaml
    service.yaml     # по желанию – конфиг сервиса (порт, пути к моделям и т.п.)
  data/
    ...
```

Главные идеи:

- **core.py**: вся содержательная логика (модель, препроцессинг, загрузка ресурсов);
- **cli.py**: точка входа для командной строки (обучение, оффлайн-предсказания и т.п.);
- **api.py**: HTTP-обёртка над тем же `core.py`.

Так легче:

- переиспользовать код;
- тестировать core независимо от API и CLI;
- изменять интерфейсы, не ломая бизнес-логику.

---

## 7. Конфигурация сервиса (порт, режимы и т.п.)

Самый простой вариант – на этапе курса **не усложнять** и задавать параметры запуска через CLI:

```bash
uv run uvicorn project.api:app --reload --host 0.0.0.0 --port 8000
```

Если хочется чуть аккуратнее, можно:

- держать `service.yaml`/`.env`;
- читать его в `api.py` или `core.py`.

Пример чтения из переменной окружения:

```python
# project/src/project/api.py
import os

from fastapi import FastAPI

PORT = int(os.getenv("SERVICE_PORT", "8000"))

app = FastAPI(title="Demo ML Service", version="0.1.0")
```

Тогда запуск:

```bash
export SERVICE_PORT=9000   # Linux/macOS
# set SERVICE_PORT=9000    # Windows (PowerShell/CMD – по-разному)
uv run uvicorn project.api:app --reload --port $SERVICE_PORT
```

Для курса обычно достаточно просто фиксированного порта 8000.

---

## 8. Обработка ошибок и валидация

FastAPI автоматически:

- валидирует входные данные по Pydantic-моделям;
- отдаёт код 422, если формат неверный.

Если требуется явно отдавать HTTP-ошибки, используем `HTTPException`:

```python
from fastapi import HTTPException


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.feature1 < 0 or request.feature2 < 0:
        raise HTTPException(
            status_code=400,
            detail="Features must be non-negative.",
        )

    model = load_model()
    y = model.predict(request.feature1, request.feature2)
    return PredictResponse(prediction=y, model_version=model.version)
```

Рекомендации:

- для **ошибок клиента** (неправильные данные, отсутствующие поля) – коды 400 / 422;
- для **внутренних ошибок** (падает модель, нет файла, и т.п.) – код 500 (можно позволить исключениям всплывать, на учебном уровне это допустимо, но стоит логировать).

---

## 9. Связка CLI и API

Типичный сценарий:

- `cli.py`:

  - `train` – обучает модель и сохраняет артефакт;
  - `predict` – делает пакетное предсказание по файлу.
- `api.py`:

  - при старте грузит **ту же** модель (из артефакта);
  - предоставляет онлайн-эндпоинт `/predict`.

Важно:

- и CLI, и API должны использовать **одни и те же функции** из `core.py` (например, `load_model()`, `predict_batch()`), чтобы не было двух разных реализаций.

---

## 10. Что мы ожидаем в ДЗ/проекте по части FastAPI

Общие ожидания (конкретные формулировки могут различаться по заданиям):

1. **Наличие FastAPI-приложения**

   - Файл типа `project/src/project/api.py` с объектом `app = FastAPI(...)`.
   - Как минимум эндпоинт `/health`.

2. **Эндпоинт /predict или аналогичный**

   - POST-метод.
   - Чётко описанный формат тела запроса (Pydantic-модель).
   - Чётко описанный формат ответа.

3. **Запуск через uv**

   - Проверка строится по схеме:

     ```bash
     git clone ...
     cd ...
     uv sync
     uv run uvicorn project.api:app --reload --port 8000
     ```

4. **Валидное поведение при ошибках**

   - Нормальная валидация входных данных.
   - Понятные сообщения об ошибках (детали не обязаны быть супер-красивыми, но должны быть информативны).

5. **Переиспользование логики**

   - Модель и вся «умная» логика не должны быть зашиты в `api.py`;
   - `api.py` только вызывает функции из `core.py`/других модулей.

---

## 11. Мини-чек-лист перед сдачей

- [ ] В репозитории есть файл с FastAPI-приложением (например, `project/src/project/api.py`).

- [ ] Объект `app = FastAPI(...)` объявлен на верхнем уровне модуля.

- [ ] Есть эндпоинт `/health`, который возвращает, что сервис жив.

- [ ] Есть хотя бы один содержательный эндпоинт (`/predict` или аналогичный) c:

  - [ ] Pydantic-моделями для запроса/ответа;
  - [ ] понятными названиями полей;
  - [ ] адекватной обработкой неверных данных.

- [ ] Всё запускается по схеме:

  ```bash
  git clone ...
  cd ...
  uv sync
  uv run uvicorn project.api:app --reload --port 8000
  ```

- [ ] Логика модели вынесена из `api.py` в отдельные модули (`core.py` и т.п.).

- [ ] Эндпоинты можно попробовать через Swagger-UI (`/docs`) и они ведут себя ожидаемо.

Если всё это выполняется, твой FastAPI-сервис соответствует базовым требованиям курса и годится как точка входа для интеграции с другими компонентами (UI, другими сервисами, нагрузочным тестированием и т.п.).

---
