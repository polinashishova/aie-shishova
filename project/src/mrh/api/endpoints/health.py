""" src/mrh/api/endpoints/health.py
Эндпоинт для проверки здоровья сервиса.
"""

from fastapi import APIRouter, status, Response
from fastapi.responses import JSONResponse
import logging

from mrh.api.schemas import HealthResponse
from mrh.api.dependencies import is_models_ready


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=['health'])

@router.get("", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse | JSONResponse:
    """
    Эндпоинт для проверки состояния сервиса.
    
    Возвращает
    -------
    HealthResponse
        Со статусом 'healthy' (HTTP 200) или 'unhealthy' (HTTP 503)
    
    Коды ответа
    -----------
    - 200 OK: Сервис здоров, модели загружены
    - 503 Service Unavailable: Сервис работает, но модели не загружены
    """

    models_ok = is_models_ready()
    
    result = HealthResponse(
        status='healthy' if models_ok else 'unhealthy',
        service='movie-recsys-hybrid',
        models_loaded=models_ok
    )

    if not models_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning('Проверка здоровья сервиса не пройдена: модели не готовы')
    else:
        logger.info('Проверка здоровья сервиса пройдена: модели загружены')
    return result