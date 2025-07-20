# backend/services/01-core/nava-logic-controller/app/models/workflow.py
"""
Workflow Models
Multi-agent workflow data models and orchestration
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class WorkflowType(str, Enum):
    """Workflow execution types"""
    SINGLE = "single"           # Single AI model
    SEQUENTIAL = "sequential"   # Sequential AI execution
    PARALLEL = "parallel"       # Parallel AI execution
    HYBRID = "hybrid"          # Mixed sequential and parallel
    DECISION_TREE = "decision_tree"  # Conditional branching

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class AgentRole(str, Enum):
    """AI Agent roles in workflow"""
    ANALYZER = "analyzer"       # Analysis and research
    GENERATOR = "generator"     # Content generation
    REVIEWER = "reviewer"       # Quality review
    SYNTHESIZER = "synthesizer" # Result combination
    VALIDATOR = "validator"     # Final validation
    COORDINATOR = "coordinator" # Workflow management

class WorkflowStep(BaseModel):
    """Individual workflow step"""
    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    agent_model: str = Field(..., description="AI model to use (gpt/claude/gemini)")
    agent_role: AgentRole = Field(..., description="Role of this agent")
    
    # Step configuration
    prompt_template: str = Field(..., description="Prompt template for this step")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for step")
    expected_output: str = Field(..., description="Expected output description")
    
    # Dependencies and flow control
    depends_on: List[str] = Field(default_factory=list, description="Step dependencies")
    parallel_group: Optional[str] = Field(None, description="Parallel execution group")
    conditional_logic: Optional[Dict[str, Any]] = Field(None, description="Conditional execution logic")
    
    # Execution settings
    timeout_seconds: int = Field(default=300, description="Step timeout")
    retry_count: int = Field(default=2, description="Retry attempts")
    quality_threshold: float = Field(default=0.7, description="Minimum quality score")
    
    # Results
    output_data: Optional[Dict[str, Any]] = Field(None, description="Step output")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Step status")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

class WorkflowTemplate(BaseModel):
    """Workflow template definition"""
    template_id: str = Field(..., description="Unique template identifier")
    template_name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    workflow_type: WorkflowType = Field(..., description="Workflow execution type")
    
    # Template configuration
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    total_estimated_time: int = Field(..., description="Estimated total time in seconds")
    complexity_level: str = Field(..., description="Complexity level (simple/medium/complex)")
    
    # Template metadata
    use_cases: List[str] = Field(default_factory=list, description="Applicable use cases")
    required_models: List[str] = Field(..., description="Required AI models")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    
    # Quality and validation
    quality_gates: List[Dict[str, Any]] = Field(default_factory=list, description="Quality checkpoints")
    success_criteria: Dict[str, Any] = Field(default_factory=dict, description="Success criteria")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)

class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    execution_id: str = Field(..., description="Unique execution identifier")
    template_id: str = Field(..., description="Template used for this execution")
    user_id: str = Field(..., description="User who initiated workflow")
    
    # Execution configuration
    workflow_type: WorkflowType = Field(..., description="Workflow type")
    steps: List[WorkflowStep] = Field(..., description="Execution steps")
    current_step: Optional[str] = Field(None, description="Currently executing step")
    
    # Input and context
    initial_input: Dict[str, Any] = Field(..., description="Initial workflow input")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context")
    session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    
    # Execution tracking
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    progress_percentage: float = Field(default=0.0, description="Completion percentage")
    completed_steps: List[str] = Field(default_factory=list, description="Completed step IDs")
    failed_steps: List[str] = Field(default_factory=list, description="Failed step IDs")
    
    # Results and output
    final_output: Optional[Dict[str, Any]] = Field(None, description="Final workflow output")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict, description="Step results")
    quality_scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores per step")
    
    # Performance metrics
    total_execution_time: Optional[float] = Field(None, description="Total execution time")
    model_usage: Dict[str, int] = Field(default_factory=dict, description="Model usage count")
    cost_estimation: Optional[float] = Field(None, description="Estimated cost")
    
    # Error handling
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error information")
    retry_attempts: int = Field(default=0, description="Number of retry attempts")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    last_updated: datetime = Field(default_factory=datetime.now)

class WorkflowResult(BaseModel):
    """Workflow execution result"""
    execution_id: str = Field(..., description="Execution identifier")
    status: WorkflowStatus = Field(..., description="Final status")
    
    # Results
    output: Dict[str, Any] = Field(..., description="Workflow output")
    summary: str = Field(..., description="Result summary")
    confidence_score: float = Field(..., description="Overall confidence")
    quality_score: float = Field(..., description="Overall quality")
    
    # Performance
    execution_time: float = Field(..., description="Total execution time")
    steps_completed: int = Field(..., description="Number of completed steps")
    steps_failed: int = Field(..., description="Number of failed steps")
    
    # Model usage
    models_used: List[str] = Field(..., description="AI models used")
    model_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Per-model performance")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    user_feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback")

class WorkflowAnalytics(BaseModel):
    """Workflow analytics and metrics"""
    template_id: str = Field(..., description="Template identifier")
    
    # Usage statistics
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    
    # Performance metrics
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    average_quality_score: float = Field(default=0.0, description="Average quality")
    average_user_satisfaction: float = Field(default=0.0, description="Average user satisfaction")
    
    # Model usage analytics
    model_usage_stats: Dict[str, int] = Field(default_factory=dict, description="Model usage statistics")
    step_failure_rates: Dict[str, float] = Field(default_factory=dict, description="Step failure rates")
    
    # Cost and efficiency
    total_cost: float = Field(default=0.0, description="Total cost")
    average_cost_per_execution: float = Field(default=0.0, description="Average cost")
    efficiency_score: float = Field(default=0.0, description="Efficiency score")
    
    # Time-based analytics
    last_updated: datetime = Field(default_factory=datetime.now)
    analytics_period: str = Field(..., description="Analytics period (daily/weekly/monthly)")

# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "research_analysis": WorkflowTemplate(
        template_id="research_analysis",
        template_name="Research & Analysis Workflow",
        description="Multi-step research and analysis with synthesis",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="initial_research",
                step_name="Initial Research",
                agent_model="gemini",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Research the topic: {topic}. Provide comprehensive background information.",
                expected_output="Detailed research findings and background information"
            ),
            WorkflowStep(
                step_id="deep_analysis",
                step_name="Deep Analysis",
                agent_model="claude",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Analyze the research findings: {research_results}. Identify key insights and patterns.",
                expected_output="Analytical insights and pattern identification",
                depends_on=["initial_research"]
            ),
            WorkflowStep(
                step_id="synthesis",
                step_name="Synthesis & Recommendations",
                agent_model="gpt",
                agent_role=AgentRole.SYNTHESIZER,
                prompt_template="Synthesize the analysis: {analysis_results}. Provide actionable recommendations.",
                expected_output="Synthesized conclusions and recommendations",
                depends_on=["deep_analysis"]
            )
        ],
        total_estimated_time=600,
        complexity_level="medium",
        required_models=["gpt", "claude", "gemini"]
    ),
    
    "content_creation": WorkflowTemplate(
        template_id="content_creation",
        template_name="Multi-Agent Content Creation",
        description="Collaborative content creation with quality review",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="outline_creation",
                step_name="Content Outline",
                agent_model="claude",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Create a detailed outline for: {content_brief}",
                expected_output="Structured content outline"
            ),
            WorkflowStep(
                step_id="content_generation",
                step_name="Content Generation",
                agent_model="gpt",
                agent_role=AgentRole.GENERATOR,
                prompt_template="Write content based on outline: {outline}",
                expected_output="Complete content draft",
                depends_on=["outline_creation"]
            ),
            WorkflowStep(
                step_id="quality_review",
                step_name="Quality Review",
                agent_model="claude",
                agent_role=AgentRole.REVIEWER,
                prompt_template="Review and improve content: {content_draft}",
                expected_output="Reviewed and improved content",
                depends_on=["content_generation"]
            )
        ],
        total_estimated_time=450,
        complexity_level="medium",
        required_models=["gpt", "claude"]
    ),
    
    "parallel_analysis": WorkflowTemplate(
        template_id="parallel_analysis",
        template_name="Parallel Multi-Perspective Analysis",
        description="Parallel analysis from multiple AI perspectives",
        workflow_type=WorkflowType.PARALLEL,
        steps=[
            WorkflowStep(
                step_id="technical_analysis",
                step_name="Technical Analysis",
                agent_model="gpt",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Provide technical analysis of: {subject}",
                expected_output="Technical perspective and insights",
                parallel_group="analysis_group"
            ),
            WorkflowStep(
                step_id="business_analysis",
                step_name="Business Analysis", 
                agent_model="claude",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Provide business analysis of: {subject}",
                expected_output="Business perspective and insights",
                parallel_group="analysis_group"
            ),
            WorkflowStep(
                step_id="creative_analysis",
                step_name="Creative Analysis",
                agent_model="gemini",
                agent_role=AgentRole.ANALYZER,
                prompt_template="Provide creative analysis of: {subject}",
                expected_output="Creative perspective and insights",
                parallel_group="analysis_group"
            ),
            WorkflowStep(
                step_id="synthesis",
                step_name="Multi-Perspective Synthesis",
                agent_model="claude",
                agent_role=AgentRole.SYNTHESIZER,
                prompt_template="Synthesize perspectives: {technical_analysis}, {business_analysis}, {creative_analysis}",
                expected_output="Integrated multi-perspective analysis",
                depends_on=["technical_analysis", "business_analysis", "creative_analysis"]
            )
        ],
        total_estimated_time=300,
        complexity_level="complex",
        required_models=["gpt", "claude", "gemini"]
    )
}

# Validation functions
def validate_workflow_step_dependencies(steps: List[WorkflowStep]) -> bool:
    """Validate that step dependencies are valid"""
    step_ids = {step.step_id for step in steps}
    
    for step in steps:
        for dependency in step.depends_on:
            if dependency not in step_ids:
                logger.error(f"Invalid dependency '{dependency}' in step '{step.step_id}'")
                return False
    
    return True

def get_workflow_template(template_id: str) -> Optional[WorkflowTemplate]:
    """Get workflow template by ID"""
    return WORKFLOW_TEMPLATES.get(template_id)

def list_workflow_templates() -> List[WorkflowTemplate]:
    """List all available workflow templates"""
    return list(WORKFLOW_TEMPLATES.values())

# Export for use in other modules
__all__ = [
    "WorkflowType",
    "WorkflowStatus", 
    "AgentRole",
    "WorkflowStep",
    "WorkflowTemplate",
    "WorkflowExecution",
    "WorkflowResult",
    "WorkflowAnalytics",
    "WORKFLOW_TEMPLATES",
    "validate_workflow_step_dependencies",
    "get_workflow_template",
    "list_workflow_templates"
]