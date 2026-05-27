""" src/mrh/api/main.py
API приложение: управление жизненным циклом приложения и его создание.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mrh.api.endpoints import health, predict
from mrh.api.dependencies import load_cb_model, load_cf_model
from mrh.utils import setup_logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения FastAPI.
    
    Выполняется при запуске и остановке сервера:
    - При старте: настройка логирования, загрузка моделей в память
    - При остановке: очистка кэша моделей, освобождение ресурсов
    
    Параметры
    ----------
    app : FastAPI
        Экземпляр приложения FastAPI
    """

    setup_logging()
    logger.info('Запуск Movie RecSys API')

    try:
        load_cb_model()
        load_cf_model()
        logger.info('Модели загружены в память')
    except Exception as e:
        logger.warning('Не удалось загрузить модели в память: %s', e)
    
    yield

    logger.info('Остановка сервера и очистка ресурсов')
    try:
        load_cb_model.cache_clear()
        load_cf_model.cache_clear()
        logger.info('Кэш моделей очищен')
    except Exception as e:
        logger.exception('Ошибка при очистке кэша моделей')
    
    logger.info('Сервер остановлен')


def create_app() -> FastAPI:
    """
    Создание и настройка экземпляра FastAPI приложения.
    
    Настраивает:
    - Метаданные приложения (название, версия, описание)
    - Lifespan менеджер для управления моделями
    - CORS middleware для кросс-доменных запросов
    - Маршруты (роутеры) для эндпоинтов
    - Корневой эндпоинт с информацией о сервисе
    
    Возвращает
    -------
    FastAPI
        Настроенный экземпляр приложения FastAPI
    """

    app = FastAPI(
        title='Movie RecSys Hybrid API',
        description='Гибридная система рекомендаций фильмов',
        version='0.1.0',
        lifespan=lifespan,
        docs_url='/docs',
        redoc_url='/redoc',
        openapi_tags=[
            {'name': 'Root', 'description': 'Базовая информация о сервисе'},
            {'name': 'health', 'description': 'Проверки доступности и здоровья сервиса'},
            {'name': 'predictions', 'description': 'Получение рекомендаций'},
        ]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*']
    )

    app.include_router(health.router)
    app.include_router(predict.router)

    @app.get('/', tags=['Root'])
    async def root() -> dict:
        """
        Корневой эндпоинт с информацией о сервисе.
        
        Возвращает
        -------
        dict
            Информация о сервисе, версии и доступных эндпоинтах
        """
        return {
            'service': 'Movie RecSys Hybrid API',
            'version': '0.1.0',
            'docs': '/docs',
            'health': '/health',
            'predict': '/predict'
        }
    
    return app