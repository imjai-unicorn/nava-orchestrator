# backend/services/01-core/nava-logic-controller/app/models/chat.py
"""
Chat Models
Comprehensive chat conversation and interaction models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Message types in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"
    NOTIFICATION = "notification"

class ConversationStatus(str, Enum):
    """Conversation status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"

class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ChatContext(BaseModel):
    """Chat context information"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    
    # User context
    user_role: Optional[str] = Field(None, description="User role/position")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    user_history: Optional[Dict[str, Any]] = Field(None, description="Relevant user history")
    
    # Session context
    platform: str = Field(default="web", description="Platform (web/mobile/api)")
    device_info: Optional[Dict[str, str]] = Field(None, description="Device information")
    location_context: Optional[Dict[str, str]] = Field(None, description="Location context")
    
    # Business context
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    department: Optional[str] = Field(None, description="Department/team")
    project_context: Optional[str] = Field(None, description="Current project context")
    
    created_at: datetime = Field(default_factory=datetime.now)

class Message(BaseModel):
    """Individual message in conversation"""
    message_id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Parent conversation ID")
    
    # Message content
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    # Message attribution
    sender_id: Optional[str] = Field(None, description="Sender identifier")
    sender_type: str = Field(..., description="Sender type (user/ai/system)")
    
    # AI-specific fields
    model_used: Optional[str] = Field(None, description="AI model used (if AI message)")
    reasoning_trace: Optional[Dict[str, Any]] = Field(None, description="AI reasoning process")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence")
    
    # Message processing
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost")
    
    # Quality and validation
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Message quality score")
    validation_status: Optional[str] = Field(None, description="Validation status")
    flags: List[str] = Field(default_factory=list, description="Content flags or warnings")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)

class ChatRequest(BaseModel):
    """Incoming chat request"""
    message: str = Field(..., min_length=1, description="User message content")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    
    # Request configuration
    preferred_model: Optional[str] = Field(None, description="Preferred AI model")
    response_mode: str = Field(default="intelligent", description="Response mode (intelligent/fast/creative)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum response tokens")
    
    # Context
    context: Optional[ChatContext] = Field(None, description="Chat context")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Request metadata
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    client_info: Optional[Dict[str, str]] = Field(None, description="Client information")

class ChatResponse(BaseModel):
    """Chat response with comprehensive metadata"""
    model_config = {"protected_namespaces": ()}
    
    # Response content
    response: str = Field(..., description="AI response content")
    message_id: str = Field(..., description="Response message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    
    # AI model information
    model_used: str = Field(..., description="AI model that generated response")
    model_version: Optional[str] = Field(None, description="Model version")
    fallback_used: bool = Field(default=False, description="Whether fallback model was used")
    
    # Response quality
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality assessment")
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safety assessment")
    
    # Decision making
    reasoning: Dict[str, Any] = Field(..., description="AI decision reasoning")
    decision_factors: List[str] = Field(default_factory=list, description="Key decision factors")
    alternative_models: List[str] = Field(default_factory=list, description="Alternative models considered")
    
    # Performance metrics
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: int = Field(..., description="Total tokens consumed")
    cost_estimate: float = Field(..., description="Estimated cost")
    
    # Response metadata
    response_type: str = Field(default="standard", description="Response type")
    content_flags: List[str] = Field(default_factory=list, description="Content flags")
    requires_followup: bool = Field(default=False, description="Whether followup is needed")
    
    # Enterprise features
    compliance_check: Optional[Dict[str, Any]] = Field(None, description="Compliance validation")
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Audit trail")
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None, description="Response expiration")

class Conversation(BaseModel):
    """Complete conversation thread"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    title: Optional[str] = Field(None, description="Conversation title")
    
    # Conversation metadata
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    context: ChatContext = Field(..., description="Conversation context")
    
    # Messages
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    message_count: int = Field(default=0, description="Total message count")
    
    # Conversation analytics
    models_used: List[str] = Field(default_factory=list, description="AI models used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, description="Total estimated cost")
    average_response_time: float = Field(default=0.0, description="Average response time")
    
    # Quality metrics
    average_quality_score: Optional[float] = Field(None, description="Average quality score")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction rating")
    conversation_rating: Optional[int] = Field(None, ge=1, le=5, description="Overall rating")
    
    # Conversation management
    tags: List[str] = Field(default_factory=list, description="Conversation tags")
    summary: Optional[str] = Field(None, description="Conversation summary")
    key_topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    archived_at: Optional[datetime] = Field(None)

class ConversationSummary(BaseModel):
    """Conversation summary for analytics"""
    conversation_id: str = Field(..., description="Conversation identifier")
    
    # Basic metrics
    message_count: int = Field(..., description="Number of messages")
    duration_minutes: float = Field(..., description="Conversation duration")
    user_messages: int = Field(..., description="User message count")
    ai_messages: int = Field(..., description="AI message count")
    
    # Quality metrics
    average_quality: float = Field(..., description="Average quality score")
    average_confidence: float = Field(..., description="Average confidence")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction")
    
    # Performance metrics
    average_response_time: float = Field(..., description="Average response time")
    total_tokens: int = Field(..., description="Total tokens used")
    total_cost: float = Field(..., description="Total cost")
    
    # Content analysis
    main_topics: List[str] = Field(..., description="Main conversation topics")
    models_used: List[str] = Field(..., description="AI models utilized")
    complexity_level: str = Field(..., description="Conversation complexity")
    
    created_at: datetime = Field(default_factory=datetime.now)

class ChatFeedback(BaseModel):
    """User feedback on chat interaction"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    message_id: str = Field(..., description="Target message ID")
    conversation_id: str = Field(..., description="Parent conversation ID")
    user_id: str = Field(..., description="User providing feedback")
    
    # Feedback content
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5 stars)")
    feedback_type: str = Field(..., description="Feedback type (helpful/not_helpful/report)")
    comment: Optional[str] = Field(None, description="Written feedback")
    
    # Detailed ratings
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5, description="Accuracy rating")
    helpfulness_rating: Optional[int] = Field(None, ge=1, le=5, description="Helpfulness rating")
    clarity_rating: Optional[int] = Field(None, ge=1, le=5, description="Clarity rating")
    
    # Issues and suggestions
    reported_issues: List[str] = Field(default_factory=list, description="Reported issues")
    improvement_suggestions: Optional[str] = Field(None, description="Improvement suggestions")
    
    # Metadata
    feedback_context: Dict[str, Any] = Field(default_factory=dict, description="Feedback context")
    created_at: datetime = Field(default_factory=datetime.now)

class ChatAnalytics(BaseModel):
    """Chat analytics and metrics"""
    analytics_id: str = Field(..., description="Analytics identifier")
    time_period: str = Field(..., description="Analytics time period")
    
    # Usage metrics
    total_conversations: int = Field(default=0, description="Total conversations")
    total_messages: int = Field(default=0, description="Total messages")
    unique_users: int = Field(default=0, description="Unique user count")
    
    # Performance metrics
    average_response_time: float = Field(default=0.0, description="Average response time")
    average_quality_score: float = Field(default=0.0, description="Average quality")
    user_satisfaction_rate: float = Field(default=0.0, description="User satisfaction rate")
    
    # Model usage
    model_usage_stats: Dict[str, int] = Field(default_factory=dict, description="Model usage statistics")
    model_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Model performance")
    
    # Cost and efficiency
    total_tokens_used: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, description="Total cost")
    cost_per_conversation: float = Field(default=0.0, description="Average cost per conversation")
    
    # User behavior
    popular_topics: List[str] = Field(default_factory=list, description="Popular conversation topics")
    peak_usage_hours: List[int] = Field(default_factory=list, description="Peak usage hours")
    
    created_at: datetime = Field(default_factory=datetime.now)

# Utility functions
def create_chat_context(user_id: str, session_id: str, **kwargs) -> ChatContext:
    """Create chat context with default values"""
    return ChatContext(
        user_id=user_id,
        session_id=session_id,
        conversation_id=kwargs.get('conversation_id', f"conv_{session_id}"),
        **kwargs
    )

def create_message(conversation_id: str, content: str, message_type: MessageType, **kwargs) -> Message:
    """Create a message with default values"""
    import uuid
    return Message(
        message_id=kwargs.get('message_id', str(uuid.uuid4())),
        conversation_id=conversation_id,
        message_type=message_type,
        content=content,
        sender_type=kwargs.get('sender_type', 'user' if message_type == MessageType.USER else 'ai'),
        **kwargs
    )

def calculate_conversation_metrics(conversation: Conversation) -> Dict[str, float]:
    """Calculate conversation metrics"""
    if not conversation.messages:
        return {}
    
    ai_messages = [m for m in conversation.messages if m.message_type == MessageType.ASSISTANT]
    
    metrics = {
        "average_confidence": sum(m.confidence_score or 0 for m in ai_messages) / len(ai_messages) if ai_messages else 0,
        "average_quality": sum(m.quality_score or 0 for m in ai_messages) / len(ai_messages) if ai_messages else 0,
        "average_response_time": sum(m.processing_time or 0 for m in ai_messages) / len(ai_messages) if ai_messages else 0,
        "total_tokens": sum(m.tokens_used or 0 for m in conversation.messages),
        "total_cost": sum(m.cost_estimate or 0 for m in conversation.messages)
    }
    
    return metrics

def validate_chat_request(request: ChatRequest) -> tuple[bool, List[str]]:
    """Validate chat request"""
    errors = []
    
    # Content validation
    if len(request.message.strip()) == 0:
        errors.append("Message content cannot be empty")
    
    if len(request.message) > 10000:  # Reasonable limit
        errors.append("Message too long (max 10000 characters)")
    
    # Context validation
    if request.context and not request.context.user_id:
        errors.append("User ID required in context")
    
    return len(errors) == 0, errors

# Export all models
__all__ = [
    "MessageType",
    "ConversationStatus", 
    "MessagePriority",
    "ChatContext",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Conversation",
    "ConversationSummary",
    "ChatFeedback",
    "ChatAnalytics",
    "create_chat_context",
    "create_message",
    "calculate_conversation_metrics",
    "validate_chat_request"
]