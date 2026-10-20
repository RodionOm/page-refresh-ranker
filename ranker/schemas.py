"""Request and response shapes. Pydantic rejects bad input with a 422 before the model sees it."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from .features import CONTENT_TYPES, INTENTS


class Page(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True)
    content_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    impressions_older_30d: int = Field(ge=0, description="earlier 30-day feature window, before prev_30d")
    impressions_prev_30d: int = Field(ge=0, description="30-day feature window immediately before the label period")
    clicks_older_30d: int = Field(ge=0)
    clicks_prev_30d: int = Field(ge=0)
    sessions_older_30d: int = Field(ge=0)
    sessions_prev_30d: int = Field(ge=0)
    content_age_days: int = Field(ge=0)
    content_type: Literal[tuple(CONTENT_TYPES)]
    main_intent: Optional[Literal[tuple(INTENTS)]] = None
    word_count: Optional[int] = Field(default=None, ge=0)
    search_volume: Optional[int] = Field(default=None, ge=0)
    competition: Optional[float] = Field(default=None, ge=0, le=1)
    cpc: Optional[float] = Field(default=None, ge=0)


class ScoreRequest(BaseModel):
    pages: list[Page] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def unique_content_ids(self):
        ids = [p.content_id for p in self.pages]
        if len(ids) != len(set(ids)):
            raise ValueError("content_id must be unique within a request")
        return self


class QueueRequest(ScoreRequest):
    k: int = Field(default=50, ge=1, le=500)
    per_client: int = Field(default=5, ge=1, le=50)


class ScoredPage(BaseModel):
    content_id: str
    client_id: str
    score: float
    band: Literal["low", "medium", "high"]
    reasons: list[str]
    review_flags: list[str]


class QueueRow(ScoredPage):
    rank: int
