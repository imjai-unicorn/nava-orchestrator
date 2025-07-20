# backend/shared/models/workflow.py
"""
NAVA Workflow Models
Comprehensive workflow and multi-agent coordination models
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid import uuid4

# ==================== ENUMS ====================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class StepStatus(Enum):
    """Individual step status"""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

class StepType(Enum):
    """Type of workflow step"""
    AI_REQUEST = "ai_request"
    DECISION = "decision"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    HYBRID = "hybrid"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

# ==================== CORE DATA CLASSES ====================

@dataclass
class WorkflowContext:
    """Workflow execution context and shared data"""
    workflow_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    organization_id: Optional[str] = None
    
    # Execution context
    variables: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    
    # AI context
    ai_models_used: List[str] = field(default_factory=list)
    total_ai_requests: int = 0
    total_cost: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_variable(self, key: str, value: Any):
        """Update workflow variable"""
        self.variables[key] = value
        self.updated_at = datetime.now()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get workflow variable"""
        return self.variables.get(key, default)
    
    def add_shared_data(self, key: str, data: Any):
        """Add data to shared context"""
        self.shared_data[key] = data
        self.updated_at = datetime.now()

@dataclass
class StepInput:
    """Input configuration for a workflow step"""
    data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    ai_model: Optional[str] = None
    prompt_template: Optional[str] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_dependency(self, step_id: str):
        """Add step dependency"""
        if step_id not in self.dependencies:
            self.dependencies.append(step_id)

@dataclass
class StepOutput:
    """Output from a workflow step"""
    data: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    status: StepStatus = StepStatus.WAITING
    error_message: Optional[str] = None
    
    # Performance metrics
    execution_time: Optional[float] = None
    ai_response_time: Optional[float] = None
    quality_score: Optional[float] = None
    cost: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepConfiguration:
    """Configuration for a workflow step"""
    retry_count: int = 3
    timeout_seconds: int = 300
    allow_parallel: bool = True
    required: bool = True
    
    # AI-specific config
    ai_temperature: Optional[float] = None
    ai_max_tokens: Optional[int] = None
    ai_fallback_models: List[str] = field(default_factory=list)
    
    # Validation config
    validate_output: bool = True
    quality_threshold: float = 0.7
    
    # Business rules
    approval_required: bool = False
    approver_roles: List[str] = field(default_factory=list)

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    name: str
    step_type: StepType
    
    # Configuration
    input_config: StepInput = field(default_factory=StepInput)
    output_config: StepOutput = field(default_factory=StepOutput)
    step_config: StepConfiguration = field(default_factory=StepConfiguration)
    
    # Execution state
    status: StepStatus = StepStatus.WAITING
    current_retry: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = str(uuid4())
    
    def can_execute(self, context: WorkflowContext) -> bool:
        """Check if step can execute based on dependencies"""
        # Check if all dependencies are completed
        # This would be implemented with workflow state checking
        return True
    
    def mark_started(self):
        """Mark step as started"""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Any = None, execution_time: float = None):
        """Mark step as completed"""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now()
        if result is not None:
            self.result_data["result"] = result
        if execution_time is not None:
            self.result_data["execution_time"] = execution_time
    
    def mark_failed(self, error: str, retry: bool = True):
        """Mark step as failed"""
        self.error_history.append({
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "retry_count": self.current_retry
        })
        
        if retry and self.current_retry < self.step_config.retry_count:
            self.status = StepStatus.RETRY
            self.current_retry += 1
        else:
            self.status = StepStatus.FAILED

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    
    # Steps and execution
    steps: List[WorkflowStep] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # Metadata
    version: str = "1.0"
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Configuration
    max_execution_time: int = 3600  # seconds
    priority: Priority = Priority.NORMAL
    tags: List[str] = field(default_factory=list)
    
    # Business rules
    requires_approval: bool = False
    approver_roles: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.workflow_id:
            self.workflow_id = str(uuid4())
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow"""
        self.steps.append(step)
        self.updated_at = datetime.now()
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_ready_steps(self, context: WorkflowContext) -> List[WorkflowStep]:
        """Get steps that are ready to execute"""
        ready_steps = []
        for step in self.steps:
            if (step.status == StepStatus.WAITING and 
                step.can_execute(context)):
                ready_steps.append(step)
        return ready_steps
    
    def is_completed(self) -> bool:
        """Check if workflow is completed"""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] 
            for step in self.steps 
            if step.step_config.required
        )
    
    def has_failed(self) -> bool:
        """Check if workflow has failed"""
        return any(
            step.status == StepStatus.FAILED 
            for step in self.steps 
            if step.step_config.required
        )

@dataclass
class WorkflowExecution:
    """Runtime workflow execution instance"""
    execution_id: str
    workflow_definition: WorkflowDefinition
    context: WorkflowContext
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    
    # Results and metrics
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid4())
        self.total_steps = len(self.workflow_definition.steps)
    
    def start(self):
        """Start workflow execution"""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        self.log_event("workflow_started", {"workflow_id": self.workflow_definition.workflow_id})
    
    def pause(self):
        """Pause workflow execution"""
        self.status = WorkflowStatus.PAUSED
        self.log_event("workflow_paused", {"current_step": self.current_step})
    
    def resume(self):
        """Resume workflow execution"""
        self.status = WorkflowStatus.RUNNING
        self.log_event("workflow_resumed", {"current_step": self.current_step})
    
    def complete(self, result: Optional[Dict[str, Any]] = None):
        """Complete workflow execution"""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        self.final_result = result
        self._update_metrics()
        self.log_event("workflow_completed", {"result": result})
    
    def fail(self, error: str):
        """Mark workflow as failed"""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.now()
        self._update_metrics()
        self.log_event("workflow_failed", {"error": error})
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log workflow event"""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        })
    
    def _update_metrics(self):
        """Update performance metrics"""
        if self.started_at and self.completed_at:
            execution_time = (self.completed_at - self.started_at).total_seconds()
            self.performance_metrics.update({
                "total_execution_time": execution_time,
                "average_step_time": execution_time / max(self.total_steps, 1),
                "success_rate": (self.completed_steps / max(self.total_steps, 1)) * 100,
                "ai_requests_count": self.context.total_ai_requests,
                "total_cost": self.context.total_cost
            })
    
    def get_progress(self) -> Dict[str, Any]:
        """Get workflow progress information"""
        progress_percentage = (self.completed_steps / max(self.total_steps, 1)) * 100
        
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "progress_percentage": round(progress_percentage, 2),
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "failed_steps": self.failed_steps,
            "current_step": self.current_step,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None
        }

# ==================== MULTI-AGENT MODELS ====================

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    quality_metrics: List[str] = field(default_factory=list)
    cost_per_request: Optional[float] = None
    average_response_time: Optional[float] = None

@dataclass
class Agent:
    """AI Agent definition"""
    agent_id: str
    name: str
    model_type: str  # gpt, claude, gemini, local
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    
    # Configuration
    base_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Performance tracking
    success_rate: float = 100.0
    average_quality: float = 0.8
    average_response_time: float = 2.0
    total_requests: int = 0
    
    # Status
    is_available: bool = True
    current_load: int = 0
    max_concurrent: int = 5
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid4())
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle task type"""
        return any(cap.name == task_type for cap in self.capabilities)
    
    def is_busy(self) -> bool:
        """Check if agent is at capacity"""
        return self.current_load >= self.max_concurrent

@dataclass
class MultiAgentTask:
    """Task that can be distributed across multiple agents"""
    task_id: str
    name: str
    description: str
    
    # Task configuration
    task_type: str
    required_capabilities: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    
    # Agent assignment
    assigned_agents: List[str] = field(default_factory=list)  # Agent IDs
    primary_agent: Optional[str] = None
    
    # Execution
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_agents: int = 3
    require_consensus: bool = False
    
    # Results
    agent_results: Dict[str, Any] = field(default_factory=dict)  # agent_id -> result
    final_result: Optional[Dict[str, Any]] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Status tracking
    status: StepStatus = StepStatus.WAITING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid4())
    
    def assign_agent(self, agent_id: str, is_primary: bool = False):
        """Assign agent to task"""
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)
        if is_primary:
            self.primary_agent = agent_id
    
    def add_result(self, agent_id: str, result: Any, quality_score: float = None):
        """Add result from an agent"""
        self.agent_results[agent_id] = result
        if quality_score is not None:
            self.quality_scores[agent_id] = quality_score

@dataclass
class WorkflowTemplate:
    """Reusable workflow template"""
    template_id: str
    name: str
    description: str
    category: str
    
    # Template definition
    step_templates: List[Dict[str, Any]] = field(default_factory=list)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    # Usage statistics
    usage_count: int = 0
    success_rate: float = 100.0
    average_execution_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.template_id:
            self.template_id = str(uuid4())
    
    def create_workflow(self, name: str, custom_config: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create workflow instance from template"""
        workflow = WorkflowDefinition(
            workflow_id=str(uuid4()),
            name=name,
            description=f"Created from template: {self.name}"
        )
        
        # Apply template configuration
        config = {**self.default_config, **(custom_config or {})}
        
        # Create steps from template
        for step_template in self.step_templates:
            step = WorkflowStep(
                step_id=str(uuid4()),
                name=step_template.get("name", "Template Step"),
                step_type=StepType(step_template.get("type", "custom"))
            )
            # Apply template configuration to step
            # ... (implementation details)
            workflow.add_step(step)
        
        return workflow

# ==================== HELPER FUNCTIONS ====================

def create_ai_request_step(
    name: str,
    ai_model: str,
    prompt_template: str,
    step_id: str = None,
    dependencies: List[str] = None,
    config: Dict[str, Any] = None
) -> WorkflowStep:
    """Helper function to create AI request step"""
    
    step = WorkflowStep(
        step_id=step_id or str(uuid4()),
        name=name,
        step_type=StepType.AI_REQUEST
    )
    
    # Configure input
    step.input_config.ai_model = ai_model
    step.input_config.prompt_template = prompt_template
    if dependencies:
        step.input_config.dependencies = dependencies
    
    # Apply custom configuration
    if config:
        for key, value in config.items():
            if hasattr(step.step_config, key):
                setattr(step.step_config, key, value)
    
    return step

def create_decision_step(
    name: str,
    decision_criteria: Dict[str, Any],
    step_id: str = None,
    dependencies: List[str] = None
) -> WorkflowStep:
    """Helper function to create decision step"""
    
    step = WorkflowStep(
        step_id=step_id or str(uuid4()),
        name=name,
        step_type=StepType.DECISION
    )
    
    step.input_config.data["decision_criteria"] = decision_criteria
    if dependencies:
        step.input_config.dependencies = dependencies
    
    return step

def create_parallel_workflow(
    name: str,
    parallel_steps: List[WorkflowStep],
    merge_step: Optional[WorkflowStep] = None
) -> WorkflowDefinition:
    """Helper function to create parallel workflow"""
    
    workflow = WorkflowDefinition(
        workflow_id=str(uuid4()),
        name=name,
        execution_mode=ExecutionMode.PARALLEL
    )
    
    # Add parallel steps
    for step in parallel_steps:
        workflow.add_step(step)
    
    # Add merge step if provided
    if merge_step:
        # Make merge step depend on all parallel steps
        merge_step.input_config.dependencies = [step.step_id for step in parallel_steps]
        workflow.add_step(merge_step)
    
    return workflow

def create_multi_agent_consensus_task(
    name: str,
    task_type: str,
    input_data: Dict[str, Any],
    required_agents: int = 3
) -> MultiAgentTask:
    """Helper function to create consensus-based multi-agent task"""
    
    task = MultiAgentTask(
        task_id=str(uuid4()),
        name=name,
        task_type=task_type,
        input_data=input_data,
        execution_mode=ExecutionMode.PARALLEL,
        max_agents=required_agents,
        require_consensus=True
    )
    
    return task

# ==================== VALIDATION FUNCTIONS ====================

def validate_workflow_definition(workflow: WorkflowDefinition) -> List[str]:
    """Validate workflow definition and return list of issues"""
    issues = []
    
    # Check basic requirements
    if not workflow.name:
        issues.append("Workflow name is required")
    
    if not workflow.steps:
        issues.append("Workflow must have at least one step")
    
    # Check step dependencies
    step_ids = {step.step_id for step in workflow.steps}
    for step in workflow.steps:
        for dep_id in step.input_config.dependencies:
            if dep_id not in step_ids:
                issues.append(f"Step {step.name} has invalid dependency: {dep_id}")
    
    # Check for circular dependencies
    # ... (implementation would include cycle detection)
    
    return issues

def estimate_workflow_execution_time(workflow: WorkflowDefinition) -> float:
    """Estimate workflow execution time in seconds"""
    if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
        # Sum all step timeouts
        return sum(step.step_config.timeout_seconds for step in workflow.steps)
    elif workflow.execution_mode == ExecutionMode.PARALLEL:
        # Maximum step timeout
        return max(step.step_config.timeout_seconds for step in workflow.steps) if workflow.steps else 0
    else:
        # Hybrid mode - estimate based on dependencies
        # ... (complex dependency analysis)
        return sum(step.step_config.timeout_seconds for step in workflow.steps) * 0.7  # Rough estimate

def calculate_workflow_cost(workflow: WorkflowDefinition, context: WorkflowContext) -> float:
    """Calculate estimated workflow execution cost"""
    total_cost = 0.0
    
    for step in workflow.steps:
        if step.step_type == StepType.AI_REQUEST:
            # Estimate based on AI model and token usage
            # This would be implemented with actual pricing models
            model = step.input_config.ai_model
            if model and "gpt" in model.lower():
                total_cost += 0.02  # Example cost per request
            elif model and "claude" in model.lower():
                total_cost += 0.025
            # ... other models
    
    return total_cost

# ==================== EXPORT ====================

__all__ = [
    # Enums
    "WorkflowStatus", "StepStatus", "StepType", "ExecutionMode", "Priority",
    
    # Core models
    "WorkflowContext", "StepInput", "StepOutput", "StepConfiguration",
    "WorkflowStep", "WorkflowDefinition", "WorkflowExecution",
    
    # Multi-agent models
    "AgentCapability", "Agent", "MultiAgentTask", "WorkflowTemplate",
    
    # Helper functions
    "create_ai_request_step", "create_decision_step", "create_parallel_workflow",
    "create_multi_agent_consensus_task",
    
    # Validation functions
    "validate_workflow_definition", "estimate_workflow_execution_time",
    "calculate_workflow_cost"
]
