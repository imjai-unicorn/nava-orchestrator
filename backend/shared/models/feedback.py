"""
Feedback Models for NAVA System
File: backend/services/shared/models/feedback.py
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from .base import TaskType

class FeedbackType(str, Enum):
    """Feedback type enumeration"""
    RATING = "rating"
    THUMBS = "thumbs"
    COMMENT = "comment"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"

class FeedbackRating(int, Enum):
    """Rating scale"""
    VERY_BAD = 1
    BAD = 2
    NEUTRAL = 3
    GOOD = 4
    VERY_GOOD = 5

class ThumbsFeedback(str, Enum):
    """Thumbs feedback"""
    UP = "up"
    DOWN = "down"

class UserFeedback(BaseModel):
    """User feedback model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str
    feedback_type: FeedbackType
    rating: Optional[FeedbackRating] = None
    thumbs: Optional[ThumbsFeedback] = None
    comment: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('comment')
    def validate_comment(cls, v, values):
        if values.get('feedback_type') == FeedbackType.COMMENT and not v:
            raise ValueError('Comment is required for comment feedback type')
        return v
    
    @validator('rating')
    def validate_rating(cls, v, values):
        if values.get('feedback_type') == FeedbackType.RATING and v is None:
            raise ValueError('Rating is required for rating feedback type')
        return v
    
    @validator('thumbs')
    def validate_thumbs(cls, v, values):
        if values.get('feedback_type') == FeedbackType.THUMBS and v is None:
            raise ValueError('Thumbs feedback is required for thumbs feedback type')
        return v

class FeedbackAnalysis(BaseModel):
    """Feedback analysis model"""
    total_feedback: int = 0
    average_rating: float = 0.0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    comment_count: int = 0
    sentiment_score: float = 0.0
    common_issues: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    @property
    def thumbs_up_rate(self) -> float:
        """Calculate thumbs up rate"""
        total = self.thumbs_up_count + self.thumbs_down_count
        return self.thumbs_up_count / total if total > 0 else 0.0

class LearningInsight(BaseModel):
    """Learning insight from feedback"""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service: str
    model: str
    task_type: TaskType
    insight_type: str
    description: str
    confidence: float
    supporting_feedback_count: int
    created_at: datetime = Field(default_factory=datetime.now)
    applied: bool = False
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
