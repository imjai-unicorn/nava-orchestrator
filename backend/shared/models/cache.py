"""
Base Models for NAVA System
File: backend/services/shared/models/base.py
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

class ServiceStatus(str, Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class AIProvider(str, Enum):
    """AI provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CACHE = "cache"

class TaskType(str, Enum):
    """Task type enumeration"""
    CONVERSATION = "conversation"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CODE = "code"
    BUSINESS = "business"
    DEEP_ANALYSIS = "deep_analysis"
    RESEARCH = "research"
    MULTIMODAL = "multimodal"

class CircuitState(str, Enum):
    """Circuit breaker state"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None

class HealthCheckResponse(BaseResponse):
    """Health check response model"""
    service: str
    status: ServiceStatus
    response_time: float
    details: Optional[Dict[str, Any]] = None

class ServiceInfo(BaseModel):
    """Service information model"""
    service: str
    provider: str
    available_models: List[str]
    default_model: str
    capabilities: List[str]
    circuit_breaker_status: Optional[Dict[str, Any]] = None

class AIRequest(BaseModel):
    """AI request model"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service: str
    model: str
    task_type: TaskType
    messages: List[Dict[str, str]]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class AIResponse(BaseModel):
    """AI response model"""
    request_id: str
    success: bool
    service: str
    model: str
    task_type: TaskType
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class DecisionContext(BaseModel):
    """Decision context model"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    enterprise_context: Dict[str, Any] = Field(default_factory=dict)
    security_level: str = "standard"
    compliance_requirements: List[str] = Field(default_factory=list)

class DecisionResult(BaseModel):
    """Decision result model"""
    selected_service: str
    selected_model: str
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    decision_time: float = 0.0
    factors: Dict[str, float] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

# Utility functions
def create_error_response(error: str, error_type: str = "GeneralError", details: Dict[str, Any] = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=error,
        error_type=error_type,
        details=details or {}
    )

def create_health_response(service: str, status: ServiceStatus, response_time: float, details: Dict[str, Any] = None) -> HealthCheckResponse:
    """Create standardized health check response"""
    return HealthCheckResponse(
        service=service,
        status=status,
        response_time=response_time,
        details=details or {}
    )

def create_ai_request(service: str, model: str, task_type: TaskType, messages: List[Dict[str, str]], **kwargs) -> AIRequest:
    """Create standardized AI request"""
    return AIRequest(
        service=service,
        model=model,
        task_type=task_type,
        messages=messages,
        **kwargs
    )

def create_ai_response(request_id: str, success: bool, service: str, model: str, task_type: TaskType, **kwargs) -> AIResponse:
    """Create standardized AI response"""
    return AIResponse(
        request_id=request_id,
        success=success,
        service=service,
        model=model,
        task_type=task_type,
        **kwargs
    )
