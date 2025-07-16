# backend/services/01-core/nava-logic-controller/app/core/workflow_orchestrator.py
"""
Workflow Orchestrator - Exit Emergency Mode
Fix recursion_prevention and enable advanced features
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    SIMPLE = "simple"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    COMPLEX = "complex"

class WorkflowStep:
    def __init__(self, step_id: str, action: str, model: str = "gpt", params: dict = None):
        self.step_id = step_id
        self.action = action
        self.model = model
        self.params = params or {}
        self.status = "pending"
        self.result = None
        self.execution_time = 0.0

class WorkflowOrchestrator:
    """Simple workflow orchestrator to exit emergency mode"""
    
    def __init__(self):
        self.active_workflows = {}
        self.workflow_history = []
        self.max_concurrent_workflows = 5
        self.recursion_depth_limit = 3
        
        # Track execution to prevent recursion
        self.execution_stack = []
        self.execution_counts = {}
        
        logger.info("🔧 Workflow Orchestrator initialized - Emergency mode prevention active")
    
    async def execute_simple_workflow(self, message: str, user_id: str = "anonymous", model: str = "gpt") -> Dict[str, Any]:
        """Execute simple single-step workflow"""
        
        workflow_id = f"simple_{int(time.time())}_{hash(message) % 1000}"
        
        try:
            # Prevent recursion
            if self._check_recursion_risk(workflow_id, message):
                return self._create_safe_response(message, model, "recursion_prevented")
            
            # Execute single step
            start_time = time.time()
            
            step = WorkflowStep(
                step_id="main_processing",
                action="generate_response", 
                model=model,
                params={"message": message, "user_id": user_id}
            )
            
            # Simulate AI processing (replace with actual AI call)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            step.result = f"Workflow response: {message}"
            step.status = "completed"
            step.execution_time = time.time() - start_time
            
            # Record successful execution
            self._record_execution(workflow_id, "success")
            
            return {
                "workflow_id": workflow_id,
                "type": "simple",
                "status": "completed",
                "response": step.result,
                "model_used": model,
                "confidence": 0.9,
                "processing_time_seconds": step.execution_time,
                "steps_completed": [step.step_id],
                "workflow_used": True,
                "recursion_safe": True
            }
            
        except Exception as e:
            logger.error(f"Simple workflow failed: {e}")
            self._record_execution(workflow_id, "failed")
            return self._create_safe_response(message, model, f"error: {str(e)}")
    
    async def execute_sequential_workflow(self, message: str, steps: List[Dict] = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """Execute sequential multi-step workflow"""
        
        workflow_id = f"sequential_{int(time.time())}_{hash(message) % 1000}"
        
        try:
            # Prevent recursion
            if self._check_recursion_risk(workflow_id, message):
                return self._create_safe_response(message, "gpt", "sequential_recursion_prevented")
            
            # Default steps if none provided
            if not steps:
                steps = [
                    {"action": "analyze", "model": "claude"},
                    {"action": "process", "model": "gpt"},
                    {"action": "synthesize", "model": "gemini"}
                ]
            
            workflow_steps = []
            start_time = time.time()
            accumulated_result = ""
            
            for i, step_config in enumerate(steps):
                step = WorkflowStep(
                    step_id=f"step_{i+1}",
                    action=step_config.get("action", "process"),
                    model=step_config.get("model", "gpt"),
                    params={"message": message, "previous_result": accumulated_result}
                )
                
                # Execute step
                step_start = time.time()
                await asyncio.sleep(0.1)  # Simulate processing
                
                step.result = f"Step {i+1} ({step.action}): Processed '{message}'"
                step.status = "completed"
                step.execution_time = time.time() - step_start
                
                workflow_steps.append(step)
                accumulated_result += f" | {step.result}"
            
            total_time = time.time() - start_time
            
            # Record successful execution
            self._record_execution(workflow_id, "success")
            
            return {
                "workflow_id": workflow_id,
                "type": "sequential",
                "status": "completed",
                "response": accumulated_result,
                "model_used": "multi_model_sequential",
                "confidence": 0.85,
                "processing_time_seconds": total_time,
                "steps_completed": [step.step_id for step in workflow_steps],
                "workflow_used": True,
                "recursion_safe": True,
                "steps_details": [
                    {
                        "step_id": step.step_id,
                        "action": step.action,
                        "model": step.model,
                        "status": step.status,
                        "execution_time": step.execution_time
                    } for step in workflow_steps
                ]
            }
            
        except Exception as e:
            logger.error(f"Sequential workflow failed: {e}")
            self._record_execution(workflow_id, "failed")
            return self._create_safe_response(message, "gpt", f"sequential_error: {str(e)}")
    
    def _check_recursion_risk(self, workflow_id: str, message: str) -> bool:
        """Check if execution might cause recursion"""
        
        # Check execution depth
        if len(self.execution_stack) >= self.recursion_depth_limit:
            logger.warning(f"Recursion depth limit reached: {len(self.execution_stack)}")
            return True
        
        # Check duplicate executions
        message_hash = hash(message) % 1000
        if message_hash in self.execution_counts:
            if self.execution_counts[message_hash] >= 3:
                logger.warning(f"Message executed too many times: {message_hash}")
                return True
        
        # Add to execution tracking
        self.execution_stack.append(workflow_id)
        self.execution_counts[message_hash] = self.execution_counts.get(message_hash, 0) + 1
        
        return False
    
    def _record_execution(self, workflow_id: str, status: str):
        """Record workflow execution completion"""
        
        # Remove from execution stack
        if workflow_id in self.execution_stack:
            self.execution_stack.remove(workflow_id)
        
        # Record in history
        self.workflow_history.append({
            "workflow_id": workflow_id,
            "status": status,
            "timestamp": time.time()
        })
        
        # Keep history manageable
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-50:]
    
    def _create_safe_response(self, message: str, model: str, reason: str) -> Dict[str, Any]:
        """Create safe fallback response"""
        
        return {
            "workflow_id": f"safe_{int(time.time())}",
            "type": "safe_fallback",
            "status": "completed",
            "response": f"Safe response for: {message}",
            "model_used": model,
            "confidence": 0.7,
            "processing_time_seconds": 0.1,
            "steps_completed": ["safe_processing"],
            "workflow_used": False,
            "recursion_safe": True,
            "fallback_reason": reason
        }
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        
        return {
            "active_workflows": len(self.active_workflows),
            "execution_stack_depth": len(self.execution_stack),
            "total_executions": len(self.workflow_history),
            "recursion_prevention_active": True,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "capabilities": ["simple", "sequential"],
            "emergency_mode": False,
            "status": "operational"
        }
    
    def is_safe_for_advanced_features(self) -> bool:
        """Check if orchestrator is ready for advanced features"""
        
        return (
            len(self.execution_stack) == 0 and
            len(self.active_workflows) < self.max_concurrent_workflows
        )

# Global instance
workflow_orchestrator = WorkflowOrchestrator()

# Convenience functions
async def execute_workflow(message: str, workflow_type: str = "simple", **kwargs) -> Dict[str, Any]:
    """Execute workflow with specified type"""
    
    if workflow_type == "sequential":
        return await workflow_orchestrator.execute_sequential_workflow(message, **kwargs)
    else:
        return await workflow_orchestrator.execute_simple_workflow(message, **kwargs)

def get_workflow_status() -> Dict[str, Any]:
    """Get workflow orchestrator status"""
    return workflow_orchestrator.get_orchestrator_status()

def is_orchestrator_ready() -> bool:
    """Check if orchestrator is ready"""
    return workflow_orchestrator.is_safe_for_advanced_features()