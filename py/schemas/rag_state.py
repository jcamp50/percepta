from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class RAGState(BaseModel):
	channel: str = Field(..., description="Twitch channel identifier")
	user: Optional[str] = Field(None, description="Username (if session-scoped)")
	last_question: Optional[str] = None
	last_citations: List[str] = []
	top_k: Optional[int] = None
	half_life_minutes: Optional[int] = None
	prefilter_limit: Optional[int] = None



