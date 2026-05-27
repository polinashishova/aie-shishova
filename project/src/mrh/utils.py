""" src/mrh/utils.py
Настройка логирования; загрузка JSON файлов; инверсия словарей.
"""

import logging
import json
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


def setup_logging(level: int | str = logging.INFO, log_dir: Path = Path('artifacts/logs'), log_filename: str = 'logs.log'):
    """
    Настройка централизованного логирования с выводом в консоль и файл.
    
    Параметры
    ----------
    level : int | str, default=logging.INFO
        Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : Path, default=Path('artifacts/logs')
        Директория для хранения файлов логов
    log_filename : str, default='logs.log'
        Имя файла лога
    
    Возвращает
    -------
    logging.Logger
        Настроенный корневой логгер
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Логирование запущено. Логи будут записываться в %s', logfile)

    return logger


JsonType = Union[dict[str, Any], list[Any], str, int, float, bool, None]

def load_json(path: Path) -> JsonType:
    """
    Загрузка JSON данных из файла.
    
    Параметры
    ----------
    path : Path
        Путь к JSON файлу
    
    Возвращает
    -------
    JsonType
        Разобранное содержимое JSON. Может быть:
        - dict: объект JSON {...}
        - list: массив JSON [...]
        - str/int/float/bool/None
    
    Исключения
    ------
    FileNotFoundError
        Если файл не существует
    json.JSONDecodeError
        Если файл содержит невалидный JSON
    OSError
        Если не удалось прочитать файл
    """
    logger.info('Загрузка JSON из %s', path)

    if not path.exists():
        logger.error('JSON файл не найден: %s', path)
        raise FileNotFoundError(f'Файл не найден: {path}')

    try:
        with path.open('r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            logger.info('JSON успешно загружен из %s', path)
            return data

    except json.JSONDecodeError:
        logger.exception('Невалидный JSON формат в %s', path)
        raise

    except OSError:
        logger.exception('Не удалось прочитать JSON файл %s', path)
        raise


def save_json(data: Any, path: Path) -> None:
    """
    Сохранение данных в JSON файл.
    
    Параметры
    ----------
    data : Any
        JSON-сериализуемые данные: dict, list, str, int, float, bool, None
    path : Path
        Путь для сохранения JSON файла
    
    Исключения
    ------
    TypeError
        Если данные не сериализуются в JSON
    OSError
        Если не удалось записать файл
    """
    logger.info('Сохранение JSON в %s', path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        logger.info('JSON успешно сохранён в %s', path)

    except TypeError:
        logger.exception('Данные не сериализуются в JSON: %s', path)
        raise

    except OSError:
        logger.exception('Не удалось записать JSON файл %s', path)
        raise


def inverse_dict(dictionary: dict) -> dict:
    """
    Инвертирует словарь: меняет местами ключи и значения.
    
    Параметры
    ----------
    dictionary : dict
        Словарь для инвертирования. Предполагается, что значения уникальны и типы 
        и ключей, и значений неизменяемы.
    
    Возвращает
    -------
    dict
        Инвертированный словарь, где ключи исходного словаря стали значениями,
        а значения - ключами.
    
    Исключения
    ------
    TypeError
        Если входной параметр не является словарем
        Если значения входного словаря не хэшируемы
    """
    if not isinstance(dictionary, dict):
        raise TypeError(f"Ожидается словарь, получен {type(dictionary).__name__}")
    
    result = {}
    for k, v in dictionary.items():
        try:
            result[v] = k
        except TypeError:
            raise TypeError(
                f"Значение {v!r} (тип {type(v).__name__}) не является хэшируемым "
                "и не может быть ключом в инвертированном словаре"
            )
    
    if len(result) != len(dictionary):
        logger.warning(
            "Инвертированный словарь имеет %d ключей вместо %d: обнаружены дубликаты значений",
            len(result), len(dictionary)
        )

    return result