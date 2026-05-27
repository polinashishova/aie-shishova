""" src/mrh/api/endpoints/predict.py
Эндпоинт для предсказания рекомендаций либо по ID фильмов, либо по ID пользователей.
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
from typing import Annotated, Optional

from sklearn.pipeline import Pipeline
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from mrh.api.schemas import PredictRequest, PredictResponse, RecommendationItem
from mrh.api.dependencies import get_cb_model_dep, get_cf_model_dep
from mrh.models import recommend_by_movieId, recommend_by_userId

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/predict', tags=['predictions'])


@router.post("", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    cb_model_data: Annotated[
        tuple[Pipeline, 
              np.ndarray,
              dict[int, int],
              dict[int, int]], 
        Depends(get_cb_model_dep)],
    cf_model_data: Annotated[
        tuple[AlternatingLeastSquares, 
              csr_matrix,
              dict[int, int],
              dict[int, int],
              dict[int, int],
              dict[int, int],
              dict[Optional[int], Optional[list[int]]]], 
        Depends(get_cf_model_dep)]
) -> PredictResponse:
    """
    Эндпоинт для получения рекомендаций фильмов.
    
    Поддерживает два типа запросов:
    - По movieIds: возвращает похожие фильмы (content-based)
    - По userIds: возвращает персонализированные рекомендации (collaborative filtering)
    
    Параметры
    ----------
    request : PredictRequest
        Тело запроса с параметрами (movieIds или userIds, k)

    Зависимости
    ------------------------------------
    cb_model_data 
        Данные контент-базированной модели (загружаются из кэша)
    cf_model_data
        Данные коллаборативной модели (загружаются из кэша)
    
    Возвращает
    -------
    PredictResponse
        Объект ответа с рекомендациями и метаинформацией
    
    Исключения
    ------
    HTTPException (404)
        Если запрошенный movieId или userId не найден в модели
    HTTPException (500)
        Если произошла внутренняя ошибка при обработке
    HTTPException (400)
        Если movieIds или userIds пуст
    """
     
    request_id = str(uuid.uuid4())[:8]
    logger.info('Запрос %s: k=%d, movieIds=%s, userIds=%s',
                request_id, request.k, request.movieIds, request.userIds)
    if request.movieIds is not None and len(request.movieIds) == 0:
        raise HTTPException(
            status_code=400,
            detail="movieIds не может быть пустым списком"
        )
    
    elif request.userIds is not None and len(request.userIds) == 0:
        raise HTTPException(
            status_code=400,
            detail="userIds не может быть пустым списком"
        )
    
    try:
        if request.movieIds is not None:
                
            cb_pipeline, movie_features, movieId_to_idx, idx_to_movieId = cb_model_data

            recs = recommend_by_movieId(
                movieIds=request.movieIds,
                movie_features=movie_features,
                movieId_to_idx=movieId_to_idx,
                idx_to_movieId=idx_to_movieId,
                k=request.k
            )

            predictions = {}
            for (rec_ids, scores), req_id in zip(recs, request.movieIds):
                predictions[req_id] = [
                    RecommendationItem(movieId=mid, score=float(score))
                    for mid, score in zip(rec_ids, scores)
                ]
            
            return PredictResponse(
                recommendations=predictions,
                model_used='content_based',
                request_id=request_id
            )
        
        elif request.userIds is not None:
            
            cf_model, user_item_matrix, movieId_to_idx, userId_to_idx, idx_to_movieId, idx_to_userId, cf_neg_movieIds = cf_model_data

            recs = recommend_by_userId(
                userIds=request.userIds, 
                als_model=cf_model, 
                user_item_matrix=user_item_matrix,
                movieId_to_idx=movieId_to_idx, 
                userId_to_idx=userId_to_idx, 
                idx_to_movieId=idx_to_movieId, 
                idx_to_userId=idx_to_userId, 
                k=request.k,
                neg_movieIds=cf_neg_movieIds
            )

            predictions = {}
            for (rec_ids, scores), req_id in zip(recs, request.userIds):
                predictions[req_id] = [
                    RecommendationItem(movieId=mid, score=float(score))
                    for mid, score in zip(rec_ids, scores)
                ]
            
            return PredictResponse(
                recommendations=predictions,
                model_used='collaborative_filtering',
                request_id=request_id
            )
        
    except KeyError as e:
        logger.warning("Запрос %s: не найден %s", request_id, e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Не найден: {e}"
        )
    except Exception as e:
        logger.exception("Запрос %s: ошибка обработки", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки запроса"
        )