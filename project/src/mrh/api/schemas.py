""" src/mrh/api/schemas.py
Схемы для запросов и ответов.
"""

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, Literal, Annotated
from datetime import datetime, timezone


class PredictRequest(BaseModel):
    """
    Запрос на получение рекомендаций фильмов.
    
    Может быть двух типов:
    - По ID фильмов (cold-start): возвращает похожие фильмы
    - По ID пользователей: возвращает персонализированные рекомендации
    ID должен быть положительным числом
    За раз можно запросить рекомендации не более, чем на 500 заданных ID
    """

    movieIds: Annotated[
        Optional[list[int]], 
        Field(
            default=None, 
            description='ID фильмов для cold-start рекомендаций (похожие фильмы)',
            examples=[[1225, 329]],
            max_length=500
        )
    ]

    userIds: Annotated[
        Optional[list[int]],
        Field(
            default=None,
            description='ID пользователей для персонализированных рекомендаций фильмов',
            examples=[[214, 154198]],
            max_length=500
        )
    ]

    k: Annotated[
        int,
        Field(
            default=10,
            description='Количество фильмов для одной рекомендации',
            ge = 1, 
            le = 100
        )
    ]

    @field_validator('movieIds', 'userIds', mode='after')
    @classmethod
    def validate_ids(cls, v):
        if v is not None:
            if any(id < 0 for id in v):
                raise ValueError('ID должен быть положительным числом')
        return v

    @model_validator(mode='after')
    def check_either_movies_or_users(self):
        """Проверка, что указан ровно один из параметров: movieIds или userIds."""
        if self.movieIds is not None and self.userIds is not None:
            raise ValueError('Требуется указать либо только movieIds, либо только userIds. Указано и то, и другое')
        elif self.movieIds is None and self.userIds is None:
            raise ValueError('Требуется указать либо movieIds, либо userIds')
        return self
    
    
class RecommendationItem(BaseModel):
    """
    Элемент рекомендации: фильм с его оценкой релевантности.
    """

    movieId: Annotated[
        int, 
        Field(description='ID рекомендованного фильма')
        ]

    score: Annotated[
        float, 
        Field(description='Оценка релевантности рекомендованного фильма (чем выше, тем лучше)')
        ]


class PredictResponse(BaseModel):
    """
    Ответ с рекомендациями фильмов.
    """

    recommendations: Annotated[
        dict[int, list[RecommendationItem]],
        Field(description='Словарь рекомендаций: {"запрошенный ID": [рекомендации]}')
    ]

    model_used: Annotated[
        Literal['content_based', 'collaborative_filtering'],
        Field(description='Тип использованной для предсказания модели')
    ]

    timestamp: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat(),
            description='Время ответа'
        )
    ]

    request_id: Annotated[
        Optional[str],
        Field(
            default=None,
            description='ID запроса'
        )
    ]


class HealthResponse(BaseModel):
    """
    Ответ на проверку здоровья сервиса.
    """

    status: Annotated[
        Literal['healthy', 'unhealthy'],
        Field(description='Статус сервиса')
    ]

    service: Annotated[
        Literal['movie-recsys-hybrid'],
        Field(
            default='movie-recsys-hybrid',
            description='Название сервиса'
        )
    ]

    models_loaded: Annotated[
        bool,
        Field(description='Флаг, загружены ли модели')
    ]

    timestamp: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat(),
            description='Время ответа'
        )
    ]