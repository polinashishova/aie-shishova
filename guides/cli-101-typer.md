# cli-101 – командные интерфейсы на Typer

## 1. Зачем нам CLI и Typer

Во многих заданиях курса (домашки и особенно проект) нужно не просто написать набор функций, а сделать **мини-приложение**, которым можно пользоваться из командной строки:

```bash
uv run python -m project.cli train --config configs/train.yaml
uv run python -m project.cli predict --input data/input.csv --output data/output.csv
```

Это называется **CLI (Command-Line Interface)** – интерфейс командной строки.

На курсе мы используем библиотеку **[Typer](https://typer.tiangolo.com/)**, потому что она:

- позволяет описывать команды в виде обычных Python-функций;
- автоматически генерирует `--help` и проверку типов аргументов;
- даёт нам единый, предсказуемый паттерн для всех студенческих CLI-приложений.

> Идея: вместо того чтобы каждый придумывал свой способ парсить аргументы `sys.argv`, мы все пользуемся Typer и придерживаемся общего шаблона.

---

## 2. Что нужно уметь до CLI на Typer

Перед тем как делать CLI на Typer, предполагается, что ты уже:

- уверенно работаешь в терминале (см. `unix-101.md`);
- понимаешь базовую структуру Python-проекта (см. `python-101.md`);
- умеешь запускать код через `uv run ...` и работаешь с зависимостями (см. `uv-101.md`);
- знаешь основы Python-функций и модулей.

Typer – просто удобный способ "обернуть" твои функции в командный интерфейс.

---

## 3. Минимальный пример приложения на Typer

### 3.1. Установка Typer (если нужно)

Если Typer ещё не добавлен в зависимости проекта:

```bash
uv add typer
```

(в курсовых репозиториях Typer может быть уже прописан в `pyproject.toml` – тогда этот шаг не нужен).

### 3.2. Самый простой пример

Создадим файл `cli.py`:

```python
# cli.py
import typer

app = typer.Typer()


@app.command()
def hello(name: str = "world"):
    """
    Простой пример команды: выводит приветствие.
    """
    typer.echo(f"Hello, {name}!")


if __name__ == "__main__":
    app()
```

Запуск:

```bash
uv run python cli.py hello
uv run python cli.py hello --name "Student"
```

Typer автоматически делает:

- парсинг аргументов;
- генерацию `--help`:

```bash
uv run python cli.py --help
uv run python cli.py hello --help
```

---

## 4. Рекомендуемая структура проекта с CLI на курсе

В курсовых репозиториях мы **не кладём** CLI «просто в корень». Мы собираем проект как пакет, чтобы удобнее импортировать код и запускать через `-m`.

Типовой вариант структуры:

```text
project/
  src/
    project/
      __init__.py
      cli.py          # точка входа CLI
      core.py         # основная логика
      io_utils.py     # функции ввода/вывода
      ...
  configs/
    train.yaml
    predict.yaml
  data/
    ...
```

Точка входа:

```python
# project/src/project/cli.py
import typer
from .core import train_model, predict_file

app = typer.Typer()


@app.command()
def train(config: str = "configs/train.yaml"):
    """
    Обучить модель по конфигу.
    """
    train_model(config_path=config)


@app.command()
def predict(input: str, output: str):
    """
    Сделать предсказания по входному файлу.
    """
    predict_file(input_path=input, output_path=output)


if __name__ == "__main__":
    app()
```

Запуск из корня репозитория:

```bash
uv run python -m project.cli train --config configs/train.yaml
uv run python -m project.cli predict --input data/input.csv --output data/output.csv
```

Обрати внимание:

- мы запускаем **модуль** `project.cli`, а не `python project/src/project/cli.py` руками;
- логика (обучение, предсказания и т.п.) лежит в других файлах (`core.py`), а CLI только «пробрасывает» параметры.

---

## 5. Команды, аргументы и опции

### 5.1. Одна команда – одна функция

Базовый паттерн:

```python
@app.command()
def my_command(
    param1: int,
    param2: str = "default",
    flag: bool = False,
):
    """
    Краткое описание команды.
    """
    ...
```

- Позиционные аргументы – без значений по умолчанию (`param1: int`).
- Опции (опциональные аргументы) – с дефолтами (`param2: str = "default"`, `flag: bool = False`).

Примеры:

```bash
uv run python -m project.cli my-command 10
uv run python -m project.cli my-command 10 --param2 "hello"
uv run python -m project.cli my-command 10 --flag
```

Typer сам:

- приведёт типы (например, `int` из строки);
- покажет ошибку, если аргументы не те.

### 5.2. Переименование аргументов

Если нужно, можно управлять именем CLI-параметра:

```python
@app.command()
def train(
    config_path: str = typer.Option("configs/train.yaml", "--config", "-c", help="Путь к конфигурационному файлу."),
):
    ...
```

Тогда вызов:

```bash
uv run python -m project.cli train --config configs/alt_train.yaml
uv run python -m project.cli train -c configs/alt_train.yaml
```

---

## 6. Подкоманды и группировка команд

Если команд становится много, логично разбить их на группы (подкоманды), например `data`, `model`, `service`.

### 6.1. Отдельные Typer-приложения

```python
# project/src/project/cli.py
import typer

from . import data_cli
from . import model_cli

app = typer.Typer()

app.add_typer(data_cli.app, name="data")
app.add_typer(model_cli.app, name="model")
```

```python
# project/src/project/data_cli.py
import typer

app = typer.Typer(help="Операции с данными.")


@app.command()
def prepare(input: str, output: str):
    """
    Подготовить датасет.
    """
    ...
```

```python
# project/src/project/model_cli.py
import typer

app = typer.Typer(help="Операции с моделью.")


@app.command()
def train(config: str = "configs/train.yaml"):
    """
    Обучить модель.
    """
    ...
```

Теперь команды выглядят так:

```bash
uv run python -m project.cli data prepare --input raw/data.csv --output data/prepared.csv
uv run python -m project.cli model train --config configs/train.yaml
```

И `--help` тоже группируется:

```bash
uv run python -m project.cli --help
uv run python -m project.cli data --help
uv run python -m project.cli model --help
```

---

## 7. Обработка ошибок и коды возврата

CLI-команда должна:

- логично реагировать на ошибки;
- возвращать ненулевой код возврата при фатальной ошибке.

### 7.1. Исключения и typer.Exit

Простой вариант – бросить исключение:

```python
@app.command()
def predict(input: str, output: str):
    if not Path(input).exists():
        raise FileNotFoundError(f"Input file not found: {input}")
    ...
```

Более аккуратный вариант – использовать `typer.Exit`:

```python
import typer
from pathlib import Path


@app.command()
def predict(input: str, output: str):
    input_path = Path(input)
    if not input_path.exists():
        typer.echo(f"[ERROR] Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    # основная логика
    ...
```

Так:

- сообщение об ошибке идёт в stderr;
- код возврата процесса – `1` (можно менять).

Это особенно полезно, если CLI потом вызывается из других скриптов/CI.

---

## 8. Что мы ожидаем в ДЗ и проекте (контракт курса)

Конкретные команды зависят от задания, но общие ожидания такие:

1. **Единая точка входа**

   - Что-то в духе: `uv run python -m project.cli ...`.
   - Или, для ДЗ: `uv run python homeworks/HW02/cli.py ...` (если структура попроще).

2. **Осмысленные команды**

   - Для проекта с ML-моделью логично иметь:

     - `train` – обучение;
     - `evaluate` – оценка;
     - `predict` – применение на новых данных;
     - (по желанию) `serve` или отдельный CLI для FastAPI-сервиса.

3. **Нормальный help**

   - Команды и параметры должны иметь внятные описания (`help=` в `typer.Option`, docstring у функции).
   - При `--help` должно быть понятно, что делает команда и какие параметры обязательны.

4. **Чёткие параметры ввода/вывода**

   - Пути к файлам/каталогам, конфиги, порты сервисов и т.п. – всё должно быть параметризуемо.
   - Жёстко зашитые пути допускаются только как разумные дефолты (`configs/train.yaml`, `data/input.csv` и т.п.).

5. **Работа через uv**

   - Проверка на курсе строится по схеме:

     ```bash
     git clone ...
     cd ...
     uv sync
     uv run python -m project.cli <команда> ...
     ```

   - Если так проект не запускается – это проблема.

---

## 9. Мини-гайд по стилю и структуре CLI

Рекомендации по стилю:

- **Названия команд** – короткие, глагольные, в нижнем регистре: `train`, `predict`, `prepare-data`, `evaluate`.
- **Аргументы** – чёткие и ожидаемые:

  - `--config`, `--input`, `--output`, `--model-path`, `--port`.
- **Логику** держать в отдельных функциях/модулях:

  - CLI должен только:

    - прочитать аргументы/опции;
    - вызвать нужную функцию из `core.py`/`service.py`;
    - обработать ошибки и вывести понятное сообщение.
- **Повторяющийся код** (логгеры, чтение конфигов) лучше вынести в отдельные вспомогательные модули.

Плохой пример:

```python
@app.command()
def train():
    # 200 строк кода обучения, чтение файлов, логика фичей внутри этой функции
    ...
```

Хороший пример:

```python
@app.command()
def train(config: str = "configs/train.yaml"):
    """
    Обучить модель по конфигу.
    """
    cfg = load_config(config)
    train_model(cfg)
```

---

## 10. Частые ошибки и как их избежать

### 10.1. Запуск без `-m` и проблемные импорты

Если запускать `python project/src/project/cli.py`, импорты вида `from .core import ...` могут не работать.

**Как надо:**

```bash
uv run python -m project.cli <команда> ...
```

### 10.2. Смешивание логики и CLI

Если вся логика зашита прямо в команду, её тяжело:

- тестировать;
- переиспользовать (например, из FastAPI-сервиса).

**Решение:** логика живёт в `core.py`/`service.py`, CLI – тонкая обёртка.

### 10.3. Жёстко зашитые пути

Плохо:

```python
@app.command()
def train():
    df = pd.read_csv("C:/Users/User/Desktop/data.csv")
```

Хорошо:

```python
@app.command()
def train(data_path: str = "data/train.csv"):
    df = pd.read_csv(data_path)
```

### 10.4. Отсутствие `--help` и описаний

Typer сам генерирует help, но если не писать docstring и `help=`, пользователю будет непонятно.

> Перед сдачей проекта/ДЗ открой `--help` у своих команд и посмотри на CLI глазами человека, который видит его впервые.

---

## 11. Чек-лист перед сдачей CLI

- [ ] В проекте есть понятная точка входа CLI (например, `project/src/project/cli.py`).
- [ ] Запуск через:

```bash
uv run python -m project.cli --help
```

работает и показывает список команд.

- [ ] Для ключевых команд (`train`, `predict`, `prepare-data` и т.п.):

  - [ ] есть внятный docstring/описание;
  - [ ] есть осмысленные аргументы/опции;
  - [ ] есть разумные значения по умолчанию.
- [ ] Основная логика вынесена из CLI-функций в отдельные модули.
- [ ] Ошибки (например, отсутствующие файлы) обрабатываются аккуратно, с понятным сообщением.
- [ ] Проект воспроизводим по схеме:

```bash
git clone ...
cd ...
uv sync
uv run python -m project.cli <команда> ...
```

Если всё из этого выполняется, твой CLI на Typer соответствует ожиданиям курса.

---
