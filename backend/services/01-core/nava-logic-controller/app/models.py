# backend/services/01-core/nava-logic-controller/app/models.py
# สร้างไฟล์ใหม่ - models หลักของ NAVA

from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

class ProcessRequest(BaseModel):
    user_input: str
    conversation_id: str
    user_id: str
    context: Optional[Dict[str, Any]] = None

class ProcessResponse(BaseModel):
    status: str
    ai_response: str
    selected_model: str
    confidence: float
    reasoning: Dict[str, Any]
    behavior_pattern: str
    task_type: str
    complexity_score: float
    processing_time_ms: int
    conversation_id: str
    user_id: str
    timestamp: str
    mode: str
    available_services: List[str]

class BehaviorAnalysisResult(BaseModel):
    primary_pattern: str
    confidence: float
    all_patterns: List[str]
    reasoning: str
    timestamp: str

class TaskClassificationResult(BaseModel):
    task_type: str
    complexity_score: float
    domain: str
    requirements: Dict[str, Any]
    timestamp: str