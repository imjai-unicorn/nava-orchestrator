# backend/services/01-core/nava-logic-controller/app/models/feedback.py
"""
Feedback Models
User feedback collection, processing, and learning integration
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FeedbackType(str, Enum):
    """Types of feedback"""
    RATING = "rating"               # Star rating or numeric score
    THUMBS = "thumbs"              # Thumbs up/down
    DETAILED = "detailed"          # Detailed written feedback
    CORRECTION = "correction"      # Correction to AI response
    SUGGESTION = "suggestion"      # Improvement suggestion
    REPORT = "report"             # Report issue/problem
    COMPARATIVE = "comparative"    # Compare multiple responses
    PREFERENCE = "preference"      # User preference indication

class FeedbackCategory(str, Enum):
    """Feedback categories"""
    ACCURACY = "accuracy"          # Factual accuracy
    HELPFULNESS = "helpfulness"    # How helpful the response was
    CLARITY = "clarity"           # Clarity of communication
    COMPLETENESS = "completeness" # Response completeness
    RELEVANCE = "relevance"       # Relevance to question
    CREATIVITY = "creativity"     # Creative quality
    SPEED = "speed"              # Response speed
    SAFETY = "safety"            # Safety concerns
    BIAS = "bias"                # Bias or fairness issues
    GENERAL = "general"          # General feedback

class FeedbackSentiment(str, Enum):
    """Feedback sentiment"""
    VERY_POSITIVE = "very_positive"    # 5 stars, very satisfied
    POSITIVE = "positive"              # 4 stars, satisfied
    NEUTRAL = "neutral"                # 3 stars, neutral
    NEGATIVE = "negative"              # 2 stars, dissatisfied
    VERY_NEGATIVE = "very_negative"    # 1 star, very dissatisfied

class FeedbackStatus(str, Enum):
    """Feedback processing status"""
    PENDING = "pending"           # Waiting to be processed
    PROCESSING = "processing"     # Currently being processed
    PROCESSED = "processed"       # Successfully processed
    INTEGRATED = "integrated"     # Integrated into learning
    REJECTED = "rejected"         # Rejected/invalid
    ESCALATED = "escalated"      # Escalated for review

class UserFeedback(BaseModel):
    """User feedback on AI responses"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    user_id: str = Field(..., description="User providing feedback")
    
    # Target information
    message_id: str = Field(..., description="Target message ID")
    conversation_id: str = Field(..., description="Target conversation ID")
    model_used: str = Field(..., description="AI model that generated response")
    
    # Feedback content
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    category: FeedbackCategory = Field(..., description="Feedback category")
    
    # Ratings and scores
    overall_rating: Optional[int] = Field(None, ge=1, le=5, description="Overall rating (1-5)")
    thumbs_rating: Optional[bool] = Field(None, description="Thumbs up (True) or down (False)")
    
    # Detailed ratings
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5, description="Accuracy rating")
    helpfulness_rating: Optional[int] = Field(None, ge=1, le=5, description="Helpfulness rating")
    clarity_rating: Optional[int] = Field(None, ge=1, le=5, description="Clarity rating")
    completeness_rating: Optional[int] = Field(None, ge=1, le=5, description="Completeness rating")
    
    # Written feedback
    comment: Optional[str] = Field(None, description="Written feedback comment")
    specific_issues: List[str] = Field(default_factory=list, description="Specific issues identified")
    suggestions: Optional[str] = Field(None, description="Improvement suggestions")
    
    # Contextual information
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context when feedback given")
    session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    
    # Processing information
    sentiment: Optional[FeedbackSentiment] = Field(None, description="Feedback sentiment")
    status: FeedbackStatus = Field(default=FeedbackStatus.PENDING, description="Processing status")
    
    # Metadata
    feedback_source: str = Field(default="user_interface", description="Source of feedback")
    is_anonymous: bool = Field(default=False, description="Whether feedback is anonymous")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = Field(None, description="When feedback was processed")

class FeedbackCorrection(BaseModel):
    """Correction feedback with specific fixes"""
    correction_id: str = Field(..., description="Unique correction identifier")
    feedback_id: str = Field(..., description="Parent feedback ID")
    
    # Correction details
    original_text: str = Field(..., description="Original AI response text")
    corrected_text: str = Field(..., description="User's corrected version")
    correction_type: str = Field(..., description="Type of correction (fact/grammar/style/logic)")
    
    # Context
    correction_reason: str = Field(..., description="Reason for correction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="User confidence in correction")
    
    # Validation
    is_verified: bool = Field(default=False, description="Whether correction has been verified")
    verified_by: Optional[str] = Field(None, description="Who verified the correction")
    verification_notes: Optional[str] = Field(None, description="Verification notes")
    
    created_at: datetime = Field(default_factory=datetime.now)

class FeedbackAggregation(BaseModel):
    """Aggregated feedback for a message/model/time period"""
    aggregation_id: str = Field(..., description="Unique aggregation identifier")
    
    # Aggregation scope
    target_type: str = Field(..., description="Type of target (message/conversation/model/period)")
    target_id: str = Field(..., description="Target identifier")
    time_period: Optional[str] = Field(None, description="Time period for aggregation")
    
    # Feedback counts
    total_feedback_count: int = Field(default=0, description="Total feedback received")
    rating_distribution: Dict[int, int] = Field(default_factory=dict, description="Rating distribution (1-5)")
    thumbs_up_count: int = Field(default=0, description="Thumbs up count")
    thumbs_down_count: int = Field(default=0, description="Thumbs down count")
    
    # Average scores
    average_overall_rating: float = Field(default=0.0, description="Average overall rating")
    average_accuracy: float = Field(default=0.0, description="Average accuracy rating")
    average_helpfulness: float = Field(default=0.0, description="Average helpfulness rating")
    average_clarity: float = Field(default=0.0, description="Average clarity rating")
    average_completeness: float = Field(default=0.0, description="Average completeness rating")
    
    # Sentiment analysis
    sentiment_distribution: Dict[FeedbackSentiment, int] = Field(default_factory=dict, description="Sentiment distribution")
    overall_sentiment: FeedbackSentiment = Field(default=FeedbackSentiment.NEUTRAL, description="Overall sentiment")
    
    # Issue analysis
    common_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Most common issues")
    improvement_themes: List[str] = Field(default_factory=list, description="Common improvement themes")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    sample_size: int = Field(default=0, description="Number of feedback items")

class FeedbackInsight(BaseModel):
    """Insights derived from feedback analysis"""
    insight_id: str = Field(..., description="Unique insight identifier")
    insight_type: str = Field(..., description="Type of insight (pattern/trend/issue/opportunity)")
    
    # Insight content
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in insight")
    
    # Data supporting insight
    supporting_data: Dict[str, Any] = Field(..., description="Data supporting the insight")
    affected_models: List[str] = Field(default_factory=list, description="AI models affected")
    affected_categories: List[FeedbackCategory] = Field(default_factory=list, description="Categories affected")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    priority: str = Field(..., description="Priority level (low/medium/high/critical)")
    estimated_impact: str = Field(..., description="Estimated impact if addressed")
    
    # Implementation
    implementation_complexity: str = Field(..., description="Implementation complexity (easy/medium/hard)")
    estimated_effort: str = Field(..., description="Estimated effort required")
    
    # Tracking
    status: str = Field(default="identified", description="Insight status (identified/planned/in_progress/implemented)")
    assigned_to: Optional[str] = Field(None, description="Team/person assigned")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class FeedbackLearning(BaseModel):
    """Learning derived from feedback for model improvement"""
    learning_id: str = Field(..., description="Unique learning identifier")
    
    # Learning source
    source_feedback_ids: List[str] = Field(..., description="Feedback IDs that contributed to learning")
    learning_category: str = Field(..., description="Category of learning")
    
    # Learning content
    pattern_identified: str = Field(..., description="Identified pattern from feedback")
    learning_rule: str = Field(..., description="Learning rule or principle")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in learning")
    
    # Application
    applicable_models: List[str] = Field(..., description="Models this learning applies to")
    applicable_contexts: List[str] = Field(default_factory=list, description="Contexts where learning applies")
    
    # Implementation
    implementation_method: str = Field(..., description="How to implement this learning")
    expected_improvement: float = Field(..., description="Expected improvement percentage")
    
    # Validation
    has_been_tested: bool = Field(default=False, description="Whether learning has been tested")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Test results if available")
    
    # Status
    status: str = Field(default="identified", description="Learning status")
    implemented_at: Optional[datetime] = Field(None, description="When learning was implemented")
    
    created_at: datetime = Field(default_factory=datetime.now)

class FeedbackAnalytics(BaseModel):
    """Analytics on feedback patterns and trends"""
    analytics_id: str = Field(..., description="Analytics identifier")
    analysis_period: str = Field(..., description="Time period analyzed")
    
    # Overall metrics
    total_feedback_received: int = Field(default=0, description="Total feedback in period")
    feedback_response_rate: float = Field(default=0.0, description="Percentage of interactions with feedback")
    average_user_satisfaction: float = Field(default=0.0, description="Average user satisfaction")
    
    # Trend analysis
    satisfaction_trend: List[float] = Field(default_factory=list, description="Satisfaction over time")
    feedback_volume_trend: List[int] = Field(default_factory=list, description="Feedback volume over time")
    issue_trend: Dict[str, List[int]] = Field(default_factory=dict, description="Issue frequency trends")
    
    # Model comparison
    model_satisfaction_comparison: Dict[str, float] = Field(default_factory=dict, description="Satisfaction by model")
    model_issue_comparison: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Issues by model")
    
    # User segmentation
    satisfaction_by_user_type: Dict[str, float] = Field(default_factory=dict, description="Satisfaction by user type")
    feedback_patterns_by_segment: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Patterns by user segment")
    
    # Actionable insights
    top_improvement_opportunities: List[str] = Field(default_factory=list, description="Top improvement opportunities")
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues requiring attention")
    success_patterns: List[str] = Field(default_factory=list, description="Patterns leading to high satisfaction")
    
    # ROI analysis
    feedback_implementation_roi: float = Field(default=0.0, description="ROI of feedback implementations")
    estimated_improvement_impact: float = Field(default=0.0, description="Estimated impact of pending improvements")
    
    created_at: datetime = Field(default_factory=datetime.now)

class FeedbackProcessor(BaseModel):
    """Feedback processing configuration and state"""
    processor_id: str = Field(..., description="Processor identifier")
    
    # Processing configuration
    auto_process_threshold: float = Field(default=0.8, description="Confidence threshold for auto-processing")
    batch_size: int = Field(default=100, description="Batch processing size")
    processing_frequency: str = Field(default="hourly", description="Processing frequency")
    
    # Learning integration
    learning_enabled: bool = Field(default=True, description="Whether to generate learning from feedback")
    learning_threshold: int = Field(default=5, description="Minimum feedback count to generate learning")
    
    # Quality control
    require_human_review: bool = Field(default=True, description="Whether critical feedback requires human review")
    escalation_triggers: List[str] = Field(default_factory=list, description="Conditions that trigger escalation")
    
    # Processing state
    last_processed_at: Optional[datetime] = Field(None, description="Last processing timestamp")
    pending_feedback_count: int = Field(default=0, description="Count of pending feedback")
    processing_errors: List[str] = Field(default_factory=list, description="Recent processing errors")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Utility functions for feedback processing
def calculate_feedback_sentiment(feedback: UserFeedback) -> FeedbackSentiment:
    """Calculate sentiment from feedback ratings and content"""
    if feedback.overall_rating:
        if feedback.overall_rating >= 5:
            return FeedbackSentiment.VERY_POSITIVE
        elif feedback.overall_rating >= 4:
            return FeedbackSentiment.POSITIVE
        elif feedback.overall_rating >= 3:
            return FeedbackSentiment.NEUTRAL
        elif feedback.overall_rating >= 2:
            return FeedbackSentiment.NEGATIVE
        else:
            return FeedbackSentiment.VERY_NEGATIVE
    
    if feedback.thumbs_rating is not None:
        return FeedbackSentiment.POSITIVE if feedback.thumbs_rating else FeedbackSentiment.NEGATIVE
    
    # Analyze comment sentiment if available
    if feedback.comment:
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "excellent", "helpful", "clear", "accurate", "useful"]
        negative_words = ["bad", "poor", "wrong", "unclear", "unhelpful", "inaccurate", "useless"]
        
        comment_lower = feedback.comment.lower()
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)
        
        if positive_count > negative_count:
            return FeedbackSentiment.POSITIVE
        elif negative_count > positive_count:
            return FeedbackSentiment.NEGATIVE
    
    return FeedbackSentiment.NEUTRAL

def validate_feedback(feedback: UserFeedback) -> tuple[bool, List[str]]:
    """Validate feedback data"""
    errors = []
    
    # Check required fields
    if not feedback.user_id:
        errors.append("User ID is required")
    
    if not feedback.message_id:
        errors.append("Message ID is required")
    
    # Validate ratings
    if feedback.overall_rating and not (1 <= feedback.overall_rating <= 5):
        errors.append("Overall rating must be between 1 and 5")
    
    # Check for meaningful content
    if feedback.feedback_type == FeedbackType.DETAILED and not feedback.comment:
        errors.append("Detailed feedback requires a comment")
    
    if feedback.feedback_type == FeedbackType.CORRECTION and not feedback.comment:
        errors.append("Correction feedback requires a comment")
    
    return len(errors) == 0, errors

def aggregate_feedback_for_message(message_id: str, feedback_list: List[UserFeedback]) -> FeedbackAggregation:
    """Aggregate feedback for a specific message"""
    message_feedback = [f for f in feedback_list if f.message_id == message_id]
    
    if not message_feedback:
        return FeedbackAggregation(
            aggregation_id=f"msg_{message_id}",
            target_type="message",
            target_id=message_id
        )
    
    # Calculate basic metrics
    total_count = len(message_feedback)
    
    # Rating distribution
    rating_dist = {}
    overall_ratings = [f.overall_rating for f in message_feedback if f.overall_rating]
    for rating in overall_ratings:
        rating_dist[rating] = rating_dist.get(rating, 0) + 1
    
    # Thumbs counts
    thumbs_up = sum(1 for f in message_feedback if f.thumbs_rating is True)
    thumbs_down = sum(1 for f in message_feedback if f.thumbs_rating is False)
    
    # Average scores
    avg_overall = sum(overall_ratings) / len(overall_ratings) if overall_ratings else 0.0
    
    accuracy_ratings = [f.accuracy_rating for f in message_feedback if f.accuracy_rating]
    avg_accuracy = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0.0
    
    helpfulness_ratings = [f.helpfulness_rating for f in message_feedback if f.helpfulness_rating]
    avg_helpfulness = sum(helpfulness_ratings) / len(helpfulness_ratings) if helpfulness_ratings else 0.0
    
    clarity_ratings = [f.clarity_rating for f in message_feedback if f.clarity_rating]
    avg_clarity = sum(clarity_ratings) / len(clarity_ratings) if clarity_ratings else 0.0
    
    completeness_ratings = [f.completeness_rating for f in message_feedback if f.completeness_rating]
    avg_completeness = sum(completeness_ratings) / len(completeness_ratings) if completeness_ratings else 0.0
    
    # Sentiment distribution
    sentiment_dist = {}
    for feedback in message_feedback:
        sentiment = calculate_feedback_sentiment(feedback)
        sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
    
    # Overall sentiment (most common)
    overall_sentiment = max(sentiment_dist.keys(), key=lambda x: sentiment_dist[x]) if sentiment_dist else FeedbackSentiment.NEUTRAL
    
    # Common issues
    all_issues = []
    for feedback in message_feedback:
        all_issues.extend(feedback.specific_issues)
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    common_issues = [
        {"issue": issue, "count": count, "percentage": count / total_count * 100}
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    
    return FeedbackAggregation(
        aggregation_id=f"msg_{message_id}",
        target_type="message",
        target_id=message_id,
        total_feedback_count=total_count,
        rating_distribution=rating_dist,
        thumbs_up_count=thumbs_up,
        thumbs_down_count=thumbs_down,
        average_overall_rating=avg_overall,
        average_accuracy=avg_accuracy,
        average_helpfulness=avg_helpfulness,
        average_clarity=avg_clarity,
        average_completeness=avg_completeness,
        sentiment_distribution=sentiment_dist,
        overall_sentiment=overall_sentiment,
        common_issues=common_issues,
        sample_size=total_count
    )

def generate_feedback_insights(aggregations: List[FeedbackAggregation], 
                             time_period: str = "weekly") -> List[FeedbackInsight]:
    """Generate insights from feedback aggregations"""
    insights = []
    
    if not aggregations:
        return insights
    
    # Low satisfaction insight
    low_satisfaction_items = [agg for agg in aggregations if agg.average_overall_rating < 3.0 and agg.sample_size >= 3]
    if low_satisfaction_items:
        insight = FeedbackInsight(
            insight_id=f"low_satisfaction_{time_period}",
            insight_type="issue",
            title="Low User Satisfaction Detected",
            description=f"Found {len(low_satisfaction_items)} items with low satisfaction scores",
            confidence=0.9,
            supporting_data={
                "affected_items": len(low_satisfaction_items),
                "average_satisfaction": sum(agg.average_overall_rating for agg in low_satisfaction_items) / len(low_satisfaction_items)
            },
            recommendations=[
                "Review low-rated responses for common patterns",
                "Implement quality improvements for identified issues",
                "Consider model retraining or prompt optimization"
            ],
            priority="high",
            estimated_impact="medium",
            implementation_complexity="medium",
            estimated_effort="2-4 weeks"
        )
        insights.append(insight)
    
    # Common issues insight
    all_issues = []
    for agg in aggregations:
        for issue in agg.common_issues:
            all_issues.append(issue["issue"])
    
    if all_issues:
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_issues and top_issues[0][1] >= 5:  # At least 5 occurrences
            insight = FeedbackInsight(
                insight_id=f"common_issues_{time_period}",
                insight_type="pattern",
                title="Common Issues Identified",
                description=f"Top issues: {', '.join([issue for issue, count in top_issues])}",
                confidence=0.8,
                supporting_data={
                    "top_issues": dict(top_issues),
                    "total_occurrences": sum(count for _, count in top_issues)
                },
                recommendations=[
                    "Address top recurring issues",
                    "Update training data or prompts",
                    "Improve validation rules"
                ],
                priority="medium",
                estimated_impact="high",
                implementation_complexity="easy",
                estimated_effort="1-2 weeks"
            )
            insights.append(insight)
    
    return insights

def create_learning_from_feedback(feedback_list: List[UserFeedback], 
                                category: FeedbackCategory) -> Optional[FeedbackLearning]:
    """Create learning rules from feedback patterns"""
    category_feedback = [f for f in feedback_list if f.category == category]
    
    if len(category_feedback) < 5:  # Need minimum feedback to generate learning
        return None
    
    # Analyze patterns in negative feedback
    negative_feedback = [f for f in category_feedback if f.overall_rating and f.overall_rating <= 2]
    
    if not negative_feedback:
        return None
    
    # Extract common issues
    common_issues = []
    for feedback in negative_feedback:
        common_issues.extend(feedback.specific_issues)
    
    issue_counts = {}
    for issue in common_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    if not issue_counts:
        return None
    
    # Find most common issue
    top_issue = max(issue_counts.keys(), key=lambda x: issue_counts[x])
    occurrence_rate = issue_counts[top_issue] / len(negative_feedback)
    
    if occurrence_rate < 0.3:  # Need at least 30% occurrence rate
        return None
    
    # Generate learning rule
    learning_rule = f"When handling {category} feedback, avoid {top_issue} which occurs in {occurrence_rate:.1%} of negative cases"
    
    # Determine applicable models
    models_with_issue = list(set(f.model_used for f in negative_feedback if top_issue in f.specific_issues))
    
    learning = FeedbackLearning(
        learning_id=f"learning_{category}_{datetime.now().strftime('%Y%m%d')}",
        source_feedback_ids=[f.feedback_id for f in negative_feedback],
        learning_category=category,
        pattern_identified=f"Common issue: {top_issue} in {category} feedback",
        learning_rule=learning_rule,
        confidence=min(occurrence_rate * 2, 1.0),  # Convert to confidence score
        applicable_models=models_with_issue,
        applicable_contexts=[category],
        implementation_method=f"Add validation rule or training data to address {top_issue}",
        expected_improvement=occurrence_rate * 0.5  # Conservative estimate
    )
    
    return learning

# Predefined feedback templates
FEEDBACK_TEMPLATES = {
    "quick_rating": {
        "type": FeedbackType.RATING,
        "categories": [FeedbackCategory.HELPFULNESS],
        "required_fields": ["overall_rating"],
        "optional_fields": ["comment"]
    },
    "detailed_assessment": {
        "type": FeedbackType.DETAILED,
        "categories": [FeedbackCategory.ACCURACY, FeedbackCategory.HELPFULNESS, 
                      FeedbackCategory.CLARITY, FeedbackCategory.COMPLETENESS],
        "required_fields": ["overall_rating", "comment"],
        "optional_fields": ["accuracy_rating", "helpfulness_rating", "clarity_rating", "completeness_rating"]
    },
    "issue_report": {
        "type": FeedbackType.REPORT,
        "categories": [FeedbackCategory.SAFETY, FeedbackCategory.BIAS, FeedbackCategory.ACCURACY],
        "required_fields": ["category", "comment", "specific_issues"],
        "optional_fields": ["suggestions"]
    },
    "thumbs_feedback": {
        "type": FeedbackType.THUMBS,
        "categories": [FeedbackCategory.GENERAL],
        "required_fields": ["thumbs_rating"],
        "optional_fields": ["comment"]
    }
}

# Export all models and functions
__all__ = [
    "FeedbackType",
    "FeedbackCategory",
    "FeedbackSentiment", 
    "FeedbackStatus",
    "UserFeedback",
    "FeedbackCorrection",
    "FeedbackAggregation",
    "FeedbackInsight",
    "FeedbackLearning",
    "FeedbackAnalytics",
    "FeedbackProcessor",
    "FEEDBACK_TEMPLATES",
    "calculate_feedback_sentiment",
    "validate_feedback",
    "aggregate_feedback_for_message",
    "generate_feedback_insights",
    "create_learning_from_feedback"
]