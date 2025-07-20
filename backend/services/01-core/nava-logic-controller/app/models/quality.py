# backend/services/01-core/nava-logic-controller/app/models/quality.py
"""
Quality Models
Quality assessment, validation, and improvement models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QualityDimension(str, Enum):
    """Quality assessment dimensions"""
    ACCURACY = "accuracy"           # Factual correctness
    COMPLETENESS = "completeness"   # Response completeness
    RELEVANCE = "relevance"         # Relevance to query
    CLARITY = "clarity"             # Clarity and readability
    COHERENCE = "coherence"         # Logical consistency
    CREATIVITY = "creativity"       # Creative value
    USEFULNESS = "usefulness"       # Practical utility
    SAFETY = "safety"              # Safety and bias check
    COMPLIANCE = "compliance"       # Enterprise compliance
    EFFICIENCY = "efficiency"       # Response efficiency

class QualityLevel(str, Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"         # 90-100%
    GOOD = "good"                  # 80-89%
    SATISFACTORY = "satisfactory"  # 70-79%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 60-69%
    POOR = "poor"                  # Below 60%

class ValidationStatus(str, Enum):
    """Validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"

class QualityScore(BaseModel):
    """Individual quality dimension score"""
    dimension: QualityDimension = Field(..., description="Quality dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in score")
    reasoning: str = Field(..., description="Explanation for the score")
    evidence: Optional[List[str]] = Field(None, description="Supporting evidence")
    
    # Metadata
    assessed_by: str = Field(..., description="Assessment method (ai/human/system)")
    assessed_at: datetime = Field(default_factory=datetime.now)

class QualityThreshold(BaseModel):
    """Quality threshold configuration"""
    dimension: QualityDimension = Field(..., description="Quality dimension")
    minimum_score: float = Field(..., ge=0.0, le=1.0, description="Minimum acceptable score")
    warning_score: float = Field(..., ge=0.0, le=1.0, description="Score that triggers warning")
    weight: float = Field(default=1.0, ge=0.0, description="Weight in overall calculation")
    is_required: bool = Field(default=True, description="Whether this dimension is required")
    
    @validator('warning_score')
    def warning_must_be_less_than_minimum(cls, v, values):
        if 'minimum_score' in values and v >= values['minimum_score']:
            raise ValueError('warning_score must be less than minimum_score')
        return v

class QualityAssessment(BaseModel):
    """Complete quality assessment for a response"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    response_id: str = Field(..., description="Response being assessed")
    user_id: Optional[str] = Field(None, description="User who made the request")
    
    # Scores per dimension
    dimension_scores: List[QualityScore] = Field(..., description="Scores for each dimension")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted overall score")
    quality_level: QualityLevel = Field(..., description="Overall quality level")
    
    # Assessment configuration
    thresholds_used: List[QualityThreshold] = Field(..., description="Thresholds applied")
    assessment_method: str = Field(..., description="Assessment method used")
    
    # Results
    passed_validation: bool = Field(..., description="Whether assessment passed")
    failed_dimensions: List[QualityDimension] = Field(default_factory=list, description="Failed dimensions")
    warning_dimensions: List[QualityDimension] = Field(default_factory=list, description="Warning dimensions")
    
    # Improvement suggestions
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended actions")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = Field(None, description="Assessment processing time")

class QualityRule(BaseModel):
    """Quality validation rule"""
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    dimension: QualityDimension = Field(..., description="Target quality dimension")
    
    # Rule configuration
    rule_type: str = Field(..., description="Rule type (threshold/pattern/ml)")
    parameters: Dict[str, Any] = Field(..., description="Rule parameters")
    severity: str = Field(..., description="Rule severity (error/warning/info)")
    
    # Rule logic
    condition: str = Field(..., description="Rule condition expression")
    error_message: str = Field(..., description="Error message if rule fails")
    suggestion: Optional[str] = Field(None, description="Improvement suggestion")
    
    # Rule metadata
    is_active: bool = Field(default=True, description="Whether rule is active")
    applies_to: List[str] = Field(default_factory=list, description="Contexts where rule applies")
    created_at: datetime = Field(default_factory=datetime.now)

class QualityGate(BaseModel):
    """Quality gate configuration"""
    gate_id: str = Field(..., description="Unique gate identifier")
    gate_name: str = Field(..., description="Gate name")
    description: str = Field(..., description="Gate description")
    
    # Gate configuration
    required_dimensions: List[QualityDimension] = Field(..., description="Required quality dimensions")
    thresholds: List[QualityThreshold] = Field(..., description="Quality thresholds")
    rules: List[QualityRule] = Field(default_factory=list, description="Validation rules")
    
    # Gate behavior
    strict_mode: bool = Field(default=False, description="Whether all dimensions must pass")
    auto_retry: bool = Field(default=False, description="Whether to auto-retry on failure")
    max_retries: int = Field(default=0, description="Maximum retry attempts")
    
    # Enterprise settings
    compliance_required: bool = Field(default=True, description="Whether compliance is required")
    audit_trail: bool = Field(default=True, description="Whether to maintain audit trail")
    
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)

class QualityImprovement(BaseModel):
    """Quality improvement suggestion"""
    improvement_id: str = Field(..., description="Unique improvement identifier")
    assessment_id: str = Field(..., description="Related assessment")
    dimension: QualityDimension = Field(..., description="Target dimension")
    
    # Improvement details
    current_score: float = Field(..., description="Current score")
    target_score: float = Field(..., description="Target score")
    improvement_type: str = Field(..., description="Type of improvement")
    
    # Suggestions
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    actions: List[Dict[str, Any]] = Field(..., description="Recommended actions")
    resources: List[str] = Field(default_factory=list, description="Helpful resources")
    
    # Implementation
    estimated_effort: str = Field(..., description="Estimated effort (low/medium/high)")
    priority: str = Field(..., description="Priority (low/medium/high)")
    
    created_at: datetime = Field(default_factory=datetime.now)

class QualityMetrics(BaseModel):
    """Quality metrics and analytics"""
    metrics_id: str = Field(..., description="Metrics identifier")
    time_period: str = Field(..., description="Time period for metrics")
    
    # Overall metrics
    total_assessments: int = Field(default=0, description="Total assessments")
    average_quality_score: float = Field(default=0.0, description="Average quality score")
    pass_rate: float = Field(default=0.0, description="Quality gate pass rate")
    
    # Dimension-specific metrics
    dimension_averages: Dict[QualityDimension, float] = Field(default_factory=dict, description="Average scores per dimension")
    dimension_trends: Dict[QualityDimension, List[float]] = Field(default_factory=dict, description="Score trends per dimension")
    
    # Model performance
    model_quality_scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores per AI model")
    improvement_rate: float = Field(default=0.0, description="Quality improvement rate")
    
    # Issue analysis
    common_failures: List[Dict[str, Any]] = Field(default_factory=list, description="Common failure patterns")
    improvement_opportunities: List[str] = Field(default_factory=list, description="Top improvement opportunities")
    
    created_at: datetime = Field(default_factory=datetime.now)

# Enterprise quality configurations
ENTERPRISE_QUALITY_THRESHOLDS = {
    QualityDimension.ACCURACY: QualityThreshold(
        dimension=QualityDimension.ACCURACY,
        minimum_score=0.80,
        warning_score=0.70,
        weight=1.5,
        is_required=True
    ),
    QualityDimension.COMPLETENESS: QualityThreshold(
        dimension=QualityDimension.COMPLETENESS,
        minimum_score=0.75,
        warning_score=0.65,
        weight=1.2,
        is_required=True
    ),
    QualityDimension.RELEVANCE: QualityThreshold(
        dimension=QualityDimension.RELEVANCE,
        minimum_score=0.80,
        warning_score=0.70,
        weight=1.3,
        is_required=True
    ),
    QualityDimension.CLARITY: QualityThreshold(
        dimension=QualityDimension.CLARITY,
        minimum_score=0.70,
        warning_score=0.60,
        weight=1.0,
        is_required=True
    ),
    QualityDimension.SAFETY: QualityThreshold(
        dimension=QualityDimension.SAFETY,
        minimum_score=0.95,
        warning_score=0.90,
        weight=2.0,
        is_required=True
    ),
    QualityDimension.COMPLIANCE: QualityThreshold(
        dimension=QualityDimension.COMPLIANCE,
        minimum_score=0.90,
        warning_score=0.85,
        weight=1.8,
        is_required=True
    )
}

# Quality gate templates
QUALITY_GATES = {
    "standard": QualityGate(
        gate_id="standard",
        gate_name="Standard Quality Gate",
        description="Standard quality requirements for general responses",
        required_dimensions=[
            QualityDimension.ACCURACY,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELEVANCE,
            QualityDimension.CLARITY
        ],
        thresholds=[
            ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.ACCURACY],
            ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.COMPLETENESS],
            ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.RELEVANCE],
            ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.CLARITY]
        ],
        strict_mode=False,
        auto_retry=True,
        max_retries=2
    ),
    
    "enterprise": QualityGate(
        gate_id="enterprise",
        gate_name="Enterprise Quality Gate",
        description="Enterprise-grade quality requirements with compliance",
        required_dimensions=[
            QualityDimension.ACCURACY,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELEVANCE,
            QualityDimension.CLARITY,
            QualityDimension.SAFETY,
            QualityDimension.COMPLIANCE
        ],
        thresholds=list(ENTERPRISE_QUALITY_THRESHOLDS.values()),
        strict_mode=True,
        auto_retry=True,
        max_retries=3,
        compliance_required=True,
        audit_trail=True
    ),
    
    "creative": QualityGate(
        gate_id="creative",
        gate_name="Creative Quality Gate",
        description="Quality requirements for creative content",
        required_dimensions=[
            QualityDimension.CREATIVITY,
            QualityDimension.COHERENCE,
            QualityDimension.CLARITY,
            QualityDimension.USEFULNESS
        ],
        thresholds=[
            QualityThreshold(
                dimension=QualityDimension.CREATIVITY,
                minimum_score=0.70,
                warning_score=0.60,
                weight=1.5
            ),
            QualityThreshold(
                dimension=QualityDimension.COHERENCE,
                minimum_score=0.75,
                warning_score=0.65,
                weight=1.2
            ),
            ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.CLARITY],
            QualityThreshold(
                dimension=QualityDimension.USEFULNESS,
                minimum_score=0.70,
                warning_score=0.60,
                weight=1.0
            )
        ],
        strict_mode=False,
        auto_retry=True,
        max_retries=2
    )
}

# Utility functions
def calculate_overall_quality_score(scores: List[QualityScore], thresholds: List[QualityThreshold]) -> float:
    """Calculate weighted overall quality score"""
    if not scores:
        return 0.0
    
    # Create threshold lookup
    threshold_map = {t.dimension: t for t in thresholds}
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for score in scores:
        threshold = threshold_map.get(score.dimension)
        weight = threshold.weight if threshold else 1.0
        
        total_weighted_score += score.score * weight
        total_weight += weight
    
    return total_weighted_score / total_weight if total_weight > 0 else 0.0

def determine_quality_level(overall_score: float) -> QualityLevel:
    """Determine quality level from overall score"""
    if overall_score >= 0.90:
        return QualityLevel.EXCELLENT
    elif overall_score >= 0.80:
        return QualityLevel.GOOD
    elif overall_score >= 0.70:
        return QualityLevel.SATISFACTORY
    elif overall_score >= 0.60:
        return QualityLevel.NEEDS_IMPROVEMENT
    else:
        return QualityLevel.POOR

def validate_against_gate(assessment: QualityAssessment, gate: QualityGate) -> tuple[bool, List[str]]:
    """Validate assessment against quality gate"""
    failures = []
    warnings = []
    
    # Check each required dimension
    dimension_scores = {score.dimension: score for score in assessment.dimension_scores}
    
    for threshold in gate.thresholds:
        if threshold.dimension not in dimension_scores:
            if threshold.is_required:
                failures.append(f"Missing required dimension: {threshold.dimension}")
            continue
        
        score = dimension_scores[threshold.dimension]
        
        if score.score < threshold.minimum_score:
            failures.append(f"{threshold.dimension}: {score.score:.2f} < {threshold.minimum_score:.2f}")
        elif score.score < threshold.warning_score:
            warnings.append(f"{threshold.dimension}: {score.score:.2f} below warning threshold {threshold.warning_score:.2f}")
    
    # In strict mode, any failure fails the gate
    if gate.strict_mode:
        passed = len(failures) == 0
    else:
        # In non-strict mode, allow some failures if overall score is good
        passed = len(failures) == 0 or (len(failures) <= 1 and assessment.overall_score >= 0.75)
    
    return passed, failures + warnings

def get_quality_gate(gate_id: str) -> Optional[QualityGate]:
    """Get quality gate by ID"""
    return QUALITY_GATES.get(gate_id)

def list_quality_gates() -> List[QualityGate]:
    """List all available quality gates"""
    return list(QUALITY_GATES.values())

def generate_improvement_suggestions(assessment: QualityAssessment) -> List[QualityImprovement]:
    """Generate improvement suggestions based on assessment"""
    suggestions = []
    
    for score in assessment.dimension_scores:
        if score.score < 0.80:  # Below good threshold
            improvement = QualityImprovement(
                improvement_id=f"{assessment.assessment_id}_{score.dimension}",
                assessment_id=assessment.assessment_id,
                dimension=score.dimension,
                current_score=score.score,
                target_score=min(score.score + 0.2, 1.0),
                improvement_type="automated_suggestion",
                suggestions=_get_dimension_suggestions(score.dimension, score.score),
                actions=_get_dimension_actions(score.dimension, score.score),
                estimated_effort="medium" if score.score < 0.60 else "low",
                priority="high" if score.score < 0.60 else "medium"
            )
            suggestions.append(improvement)
    
    return suggestions

def _get_dimension_suggestions(dimension: QualityDimension, score: float) -> List[str]:
    """Get improvement suggestions for specific dimension"""
    suggestions_map = {
        QualityDimension.ACCURACY: [
            "Verify facts with reliable sources",
            "Cross-check information for consistency",
            "Use more specific and precise language",
            "Include data sources and references"
        ],
        QualityDimension.COMPLETENESS: [
            "Address all aspects of the question",
            "Provide more comprehensive coverage",
            "Include relevant examples and details",
            "Ensure no important points are missed"
        ],
        QualityDimension.RELEVANCE: [
            "Focus more directly on the user's question",
            "Remove tangential information",
            "Prioritize the most important aspects",
            "Align response with user's context"
        ],
        QualityDimension.CLARITY: [
            "Use simpler, clearer language",
            "Improve sentence structure and flow",
            "Add better organization and headings",
            "Define technical terms and jargon"
        ],
        QualityDimension.COHERENCE: [
            "Improve logical flow between ideas",
            "Use better transitions and connections",
            "Ensure consistent argumentation",
            "Remove contradictory statements"
        ],
        QualityDimension.CREATIVITY: [
            "Add more original insights and perspectives",
            "Include innovative approaches or solutions",
            "Use creative examples and analogies",
            "Think outside conventional frameworks"
        ],
        QualityDimension.USEFULNESS: [
            "Provide more actionable recommendations",
            "Include practical implementation steps",
            "Add real-world applications",
            "Focus on user's immediate needs"
        ],
        QualityDimension.SAFETY: [
            "Review for potential harmful content",
            "Check for bias and fairness",
            "Ensure appropriate disclaimers",
            "Verify safety of any recommendations"
        ],
        QualityDimension.COMPLIANCE: [
            "Review against enterprise policies",
            "Ensure regulatory compliance",
            "Check data privacy requirements",
            "Verify ethical guidelines adherence"
        ]
    }
    
    return suggestions_map.get(dimension, ["General improvement needed"])

def _get_dimension_actions(dimension: QualityDimension, score: float) -> List[Dict[str, Any]]:
    """Get specific actions for dimension improvement"""
    actions = []
    
    if score < 0.60:  # Poor - needs major improvement
        actions.append({
            "action": "major_revision",
            "description": f"Major revision needed for {dimension}",
            "priority": "high",
            "effort": "high"
        })
    elif score < 0.75:  # Needs improvement
        actions.append({
            "action": "targeted_improvement",
            "description": f"Targeted improvement for {dimension}",
            "priority": "medium", 
            "effort": "medium"
        })
    else:  # Minor improvement
        actions.append({
            "action": "minor_enhancement",
            "description": f"Minor enhancement for {dimension}",
            "priority": "low",
            "effort": "low"
        })
    
    return actions

# Quality assessment templates
QUALITY_ASSESSMENT_TEMPLATES = {
    "comprehensive": {
        "dimensions": [
            QualityDimension.ACCURACY,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELEVANCE,
            QualityDimension.CLARITY,
            QualityDimension.COHERENCE,
            QualityDimension.USEFULNESS,
            QualityDimension.SAFETY,
            QualityDimension.COMPLIANCE
        ],
        "method": "ai_assisted_comprehensive"
    },
    "standard": {
        "dimensions": [
            QualityDimension.ACCURACY,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELEVANCE,
            QualityDimension.CLARITY
        ],
        "method": "ai_assisted_standard"
    },
    "creative": {
        "dimensions": [
            QualityDimension.CREATIVITY,
            QualityDimension.COHERENCE,
            QualityDimension.CLARITY,
            QualityDimension.USEFULNESS
        ],
        "method": "ai_assisted_creative"
    },
    "enterprise": {
        "dimensions": [
            QualityDimension.ACCURACY,
            QualityDimension.COMPLETENESS,
            QualityDimension.SAFETY,
            QualityDimension.COMPLIANCE
        ],
        "method": "enterprise_validation"
    }
}

# Export for use in other modules
__all__ = [
    "QualityDimension",
    "QualityLevel",
    "ValidationStatus",
    "QualityScore",
    "QualityThreshold", 
    "QualityAssessment",
    "QualityRule",
    "QualityGate",
    "QualityImprovement",
    "QualityMetrics",
    "ENTERPRISE_QUALITY_THRESHOLDS",
    "QUALITY_GATES",
    "QUALITY_ASSESSMENT_TEMPLATES",
    "calculate_overall_quality_score",
    "determine_quality_level",
    "validate_against_gate",
    "get_quality_gate",
    "list_quality_gates",
    "generate_improvement_suggestions"
]