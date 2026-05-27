"""
Скрипт для запуска API приложения.

Пример использования:
cd project
python scripts/run_api.py
"""

import argparse
import sys
from pathlib import Path
from mrh.config import APP_HOST, APP_PORT

import uvicorn


def main():
    """
    Запуск UVicorn сервера для Movie RecSys API.
    
    Поддерживает аргументы командной строки для настройки хоста, порта,
    режима разработки и количества воркеров.
    """
    
    parser = argparse.ArgumentParser(description="Запуск API сервера рекомендаций фильмов")
    parser.add_argument("--host", default=APP_HOST, help="Хост для сервера")
    parser.add_argument("--port", type=int, default=APP_PORT, help="Порт для сервера")
    parser.add_argument("--reload", action="store_true", help="Автоперезагрузка при изменении кода (только для разработки!)")
    parser.add_argument("--workers", type=int, default=1, help="Количество воркеров (>=1 для продакшена)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    display_host = "localhost" if args.host in ["0.0.0.0", ""] else args.host

    print(f"Запуск сервера на http://{display_host}:{args.port}")
    print(f"Swagger UI:       http://{display_host}:{args.port}/docs")
    print(f"Health Check:     http://{display_host}:{args.port}/health")
    print(f"Режим reload:    {'ВКЛ' if args.reload else 'ВЫКЛ'}")

    uvicorn.run(
        "mrh.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()