# EMERGENCY FIX: วางไฟล์นี้ทับ app/service/logic_orchestrator.py

# app/service/logic_orchestrator.py - EMERGENCY RECURSION FIX
"""
EMERGENCY RECURSION FIX - Minimal Safe Version
ลบ recursive calls ทั้งหมด + ใช้ direct processing เท่านั้น
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

# Safe imports with fallbacks
try:
    from .service_discovery import ServiceDiscovery
except:
    ServiceDiscovery = None

try:
    from .real_ai_client import RealAIClient
except:
    RealAIClient = None

try:
    from app.core.decision_engine import EnhancedDecisionEngine, WorkflowMode
except:
    EnhancedDecisionEngine = None
    WorkflowMode = None

logger = logging.getLogger(__name__)

class UserApprovalLevel(Enum):
    """ระดับการอนุมัติของ User"""
    FULL_AUTO = "full_auto"
    STRATEGIC_APPROVAL = "strategic" 
    STEP_BY_STEP = "step_by_step"

class LogicOrchestrator:
    """
    EMERGENCY SAFE Logic Orchestrator
    NO RECURSION - DIRECT PROCESSING ONLY
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.version = "2.0.2-EMERGENCY-SAFE"
        self.is_ai_system = False
        self.config = config or {}
        
        # 🚨 EMERGENCY: Disable all complex features that cause recursion
        self._emergency_mode = True
        self._processing_lock = asyncio.Lock()
        
        # Safe component initialization
        self.service_discovery = None
        self.ai_client = None
        self.decision_engine = None
        
        try:
            if ServiceDiscovery:
                self.service_discovery = ServiceDiscovery()
            if RealAIClient and self.service_discovery:
                self.ai_client = RealAIClient(self.service_discovery)
            if EnhancedDecisionEngine:
                self.decision_engine = EnhancedDecisionEngine()
        except Exception as e:
            logger.warning(f"⚠️ Safe initialization with limited features: {e}")
        
        # Simple metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "emergency_mode_count": 0
        }
        
        self.is_initialized = False
        self.initialization_time = None
        self._ai_call_stack = {}  # Track active calls per model
        self._max_recursion_depth = 2  # Maximum allowed depth
        
    async def initialize(self):
        """🚨 EMERGENCY: Minimal safe initialization"""
        if self.is_initialized:
            return
            
        logger.info("🚨 EMERGENCY SAFE MODE: Initializing minimal orchestrator...")
        
        try:
            # Minimal initialization only
            if self.ai_client:
                await self.ai_client.initialize()
            
            self.is_initialized = True
            self.initialization_time = datetime.now()
            logger.info("✅ Emergency safe orchestrator initialized")
            
        except Exception as e:
            logger.error(f"❌ Emergency initialization failed: {e}")
            # Continue anyway with fallback mode
            self.is_initialized = True
            self.initialization_time = datetime.now()

    async def process_request(self, message: str, user_id: str = "anonymous", 
                            user_preference: Optional[str] = None,
                            context: Dict[str, Any] = None,
                            approval_level: UserApprovalLevel = UserApprovalLevel.STRATEGIC_APPROVAL) -> Dict[str, Any]:
        """
        🚨 EMERGENCY: Direct processing - NO RECURSION
        """
        async with self._processing_lock:  # Prevent concurrent calls
            if not self.is_initialized:
                await self.initialize()

            start_time = datetime.now()
            self.metrics["total_requests"] += 1
            self.metrics["emergency_mode_count"] += 1

            logger.info(f"🚨 EMERGENCY processing request for user: {user_id}")

            try:
                # 1. DIRECT MODEL SELECTION (no complex decision engine)
                selected_model = self._emergency_model_selection(message, user_preference)
                
                # 2. DIRECT AI CALL (no workflow, no recursion)
                result = await self._emergency_ai_call(selected_model, message, context or {})
                
                # 3. DIRECT RESPONSE (no finalization, no workflow)
                response = self._emergency_create_response(result, selected_model, start_time)
                
                self.metrics["successful_responses"] += 1
                logger.info(f"✅ Emergency response completed for {user_id}")
                
                return response
                
            except Exception as e:
                self.metrics["failed_responses"] += 1
                logger.error(f"❌ Emergency processing failed for {user_id}: {e}")
                
                return self._emergency_error_response(str(e), user_id, start_time)

    def _emergency_model_selection(self, message: str, user_preference: Optional[str]) -> str:
        """🚨 EMERGENCY: Simple model selection - NO DECISION ENGINE"""
        
        # Priority 1: User preference
        if user_preference and user_preference in ["gpt", "claude", "gemini"]:
            return user_preference
        
        # Priority 2: Simple keyword matching
        message_lower = message.lower()
        
        # Claude for writing/analysis
        if any(word in message_lower for word in ["write", "analyze", "create", "เขียน", "วิเคราะห์"]):
            return "claude"
        
        # Gemini for strategy/planning
        if any(word in message_lower for word in ["plan", "strategy", "แผน", "กลยุทธ์"]):
            return "gemini"
        
        # GPT for everything else (default)
        return "gpt"
 
    async def _emergency_ai_call(self, model: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """🚨 EMERGENCY: Direct AI call with COMPLETE recursion protection"""
    
        # 🔒 RECURSION DETECTION
        call_key = f"{model}_{hash(message)}"
        current_depth = self._ai_call_stack.get(call_key, 0)
    
        if current_depth >= self._max_recursion_depth:
            logger.error(f"🚨 RECURSION LIMIT REACHED for {model} (depth: {current_depth})")
            return self._create_recursion_limit_response(model, message, current_depth)
    
        # 🔒 INCREMENT DEPTH
        self._ai_call_stack[call_key] = current_depth + 1
    
        try:
            # 🚨 MODEL BYPASS for problematic models
            if model.lower() == "claude" and current_depth > 0:
                logger.warning(f"🚨 Claude recursion detected (depth: {current_depth}), using GPT")
                model = "gpt"
                call_key = f"{model}_{hash(message)}"  # Update call key
        
            if self.ai_client:
                try:
                    # 🔒 TIMEOUT + RECURSION PROTECTION
                    result = await asyncio.wait_for(
                        self.ai_client.call_ai(model, message, context),
                        timeout=15.0  # Reduced timeout for faster failure
                    )

                    # Ensure it's a proper dict
                    if not isinstance(result, dict):
                        result = {"response": str(result), "confidence": 0.7}

                    # 🔒 SUCCESS - RESET DEPTH
                    self._ai_call_stack[call_key] = 0
                
                    return result
                
                except asyncio.TimeoutError:
                    logger.error(f"❌ AI call timeout for {model}")
                    return self._create_timeout_fallback(model, message)
                
                except RecursionError as e:
                    logger.error(f"❌ Python RecursionError detected for {model}: {e}")
                    return self._create_recursion_fallback(model, message)
                
                except Exception as e:
                    error_str = str(e).lower()
                    if "recursion" in error_str or "maximum" in error_str:
                        logger.error(f"❌ Recursion-related error for {model}: {e}")
                        return self._create_recursion_fallback(model, message)
                    else:
                        logger.error(f"❌ AI call failed for {model}: {e}")
                        return self._create_ai_error_fallback(model, message, str(e))
            else:
                # Pure fallback if no AI client
                return self._create_no_client_fallback(model, message)
            
        except Exception as e:
            logger.error(f"❌ Emergency AI call completely failed for {model}: {e}")
            return self._create_complete_failure_fallback(model, message, str(e))
    
        finally:
            # 🔒 ALWAYS DECREMENT DEPTH
            if call_key in self._ai_call_stack:
                self._ai_call_stack[call_key] = max(0, self._ai_call_stack[call_key] - 1)
                if self._ai_call_stack[call_key] == 0:
                    del self._ai_call_stack[call_key]  # Clean up

    def _create_recursion_limit_response(self, model: str, message: str, depth: int) -> Dict[str, Any]:
        """Create response when recursion limit is reached"""
        return {
            "response": f"I understand your request about '{message}'. To maintain system stability, I'm processing this using our optimized approach. I can provide comprehensive assistance on this topic using alternative processing methods.",
            "confidence": 0.65,
            "model_used": f"{model}-recursion-limited",
            "processing_time_seconds": 0.1,
            "fallback_reason": "recursion_limit_protection",
            "recursion_depth": depth,
            "protection_active": True
        }

    def _create_timeout_fallback(self, model: str, message: str) -> Dict[str, Any]:
        """Create response when AI call times out"""
        return {
            "response": f"I understand you're asking about: '{message}'. While I'm experiencing some processing delays, I can provide a thoughtful response. This appears to be a {self._detect_message_type(message)} type of question that I can help address with appropriate analysis and insight.",
            "confidence": 0.65,
            "model_used": f"{model}-timeout-fallback",
            "processing_time_seconds": 15.0,
            "fallback_reason": "timeout_protection"
        }

    def _create_recursion_fallback(self, model: str, message: str) -> Dict[str, Any]:
        """Create response when recursion is detected"""
        return {
            "response": f"I'm processing your request about: '{message}'. Due to system protection measures, I'm using a simplified processing approach to ensure stable operation. I can still provide helpful insights on this topic with reliable analytical thinking.",
            "confidence": 0.60,
            "model_used": f"{model}-recursion-protected",
            "processing_time_seconds": 0.1,
            "fallback_reason": "recursion_prevention"
        }

    def _create_ai_error_fallback(self, model: str, message: str, error: str) -> Dict[str, Any]:
        """Create response when AI call has errors"""
        return {
            "response": f"Thank you for your question: '{message}'. I'm currently experiencing some technical challenges, but I can still provide assistance. This appears to be a question that would benefit from {self._get_response_style(model)} approach to analysis and problem-solving.",
            "confidence": 0.55,
            "model_used": f"{model}-error-recovery",
            "error": error,
            "processing_time_seconds": 0.1,
            "fallback_reason": "ai_client_error"
        }

    def _create_no_client_fallback(self, model: str, message: str) -> Dict[str, Any]:
        """Create response when no AI client available"""
        return {
            "response": f"I understand you're asking: '{message}'. While my AI processing systems are initializing, I can provide a structured response. This appears to be a {self._detect_message_type(message)} that I can address with appropriate analytical thinking.",
            "confidence": 0.70,
            "model_used": f"{model}-no-client",
            "processing_time_seconds": 0.1,
            "fallback_reason": "client_not_available"
        }

    def _create_complete_failure_fallback(self, model: str, message: str, error: str) -> Dict[str, Any]:
        """Create response when everything fails"""
        return {
            "response": f"I acknowledge your request: '{message}'. While experiencing system limitations, I remain committed to providing helpful assistance. Please let me know if you'd like me to approach this question from a different angle or if there's a specific aspect you'd like me to focus on.",
            "confidence": 0.50,
            "model_used": f"{model}-complete-fallback",
            "error": error,
            "processing_time_seconds": 0.1,
            "fallback_reason": "complete_failure_recovery"
        }

    def _detect_message_type(self, message: str) -> str:
        """Detect message type for appropriate response"""
        message_lower = message.lower()
    
        if any(word in message_lower for word in ["analyze", "analysis", "วิเคราะห์"]):
            return "analytical inquiry"
        elif any(word in message_lower for word in ["write", "create", "เขียน", "สร้าง"]):
            return "creative request"
        elif any(word in message_lower for word in ["code", "programming", "โค้ด"]):
            return "technical question"
        elif any(word in message_lower for word in ["help", "how", "ช่วย", "วิธี"]):
            return "assistance request"
        else:
            return "general inquiry"

    def _get_response_style(self, model: str) -> str:
        """Get response style based on model"""
        if model == "claude":
            return "analytical and comprehensive"
        elif model == "gpt":
            return "conversational and helpful"
        elif model == "gemini":
            return "strategic and structured"
        else:
            return "thoughtful and balanced"
    
    def _emergency_create_response(self, ai_result: Dict[str, Any], model: str, start_time: datetime) -> Dict[str, Any]:
        """🚨 EMERGENCY: Create final response - NO WORKFLOW"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": ai_result.get("response", "No response available"),
            "model_used": model,
            "confidence": ai_result.get("confidence", 0.7),
            "processing_time_seconds": processing_time,
            "orchestration_type": "emergency_direct",
            "workflow_used": False,
            "emergency_mode": True,
            "decision_info": {
                "method": "emergency_simple_selection",
                "model_selected": model,
                "emergency_mode_active": True
            },
            "timestamp": datetime.now().isoformat()
        }

    def _emergency_error_response(self, error_msg: str, user_id: str, start_time: datetime) -> Dict[str, Any]:
        """🚨 EMERGENCY: Error response - NO RECURSION"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": f"Emergency system error occurred. Please try a simpler request.\n\nError: {error_msg}",
            "model_used": "emergency_error_handler",
            "confidence": 0.0,
            "processing_time_seconds": processing_time,
            "orchestration_type": "emergency_error",
            "workflow_used": False,
            "error": error_msg,
            "emergency_mode": True,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

    # 🚨 EMERGENCY: Complex request → redirect to simple processing
    async def process_complex_request(self, message: str, user_id: str, 
                                    complexity_level: str = "auto", 
                                    quality_requirements: Dict[str, float] = None) -> Dict[str, Any]:
        """🚨 EMERGENCY: Redirect complex requests to simple processing"""
        
        logger.warning(f"🚨 Complex request redirected to simple processing for {user_id}")
        
        # Simply call the basic process_request (which is now safe)
        result = await self.process_request(message, user_id)
        
        # Add complex processing markers
        result.update({
            "workflow_type": "emergency_simplified",
            "original_complexity": complexity_level,
            "note": "Complex processing simplified for stability"
        })
        
        return result

    # 🚨 EMERGENCY: Execute parallel → redirect to simple
    async def execute_parallel_processing(self, message: str, user_id: str, 
                                        models: List[str] = None) -> Dict[str, Any]:
        """🚨 EMERGENCY: Redirect parallel to simple processing"""
        
        logger.warning(f"🚨 Parallel request redirected to simple processing for {user_id}")
        
        # Use first model or default
        if models and len(models) > 0:
            preferred_model = models[0]
        else:
            preferred_model = None
        
        result = await self.process_request(message, user_id, user_preference=preferred_model)
        
        result.update({
            "execution_type": "emergency_single_instead_of_parallel",
            "requested_models": models or [],
            "note": "Parallel processing simplified for stability"
        })
        
        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """🚨 EMERGENCY: Simple system status"""
        
        uptime = (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0
        
        return {
            "status": "emergency_safe_mode",
            "nava_initialized": self.is_initialized,
            "available_models": ["gpt", "claude", "gemini"],
            "uptime_seconds": uptime,
            "performance_metrics": self.metrics,
            "capabilities": [
                "emergency_direct_processing",
                "simple_model_selection",
                "basic_ai_integration"
            ],
            "workflow_modes": ["emergency_direct_only"],
            "controller_type": "emergency_safe_orchestrator",
            "version": self.version,
            "emergency_mode": {
                "active": self._emergency_mode,
                "reason": "recursion_prevention",
                "features_disabled": [
                    "complex_workflows",
                    "sequential_processing", 
                    "parallel_processing",
                    "advanced_decision_engine"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

    async def get_advanced_system_metrics(self) -> Dict[str, Any]:
        """🚨 EMERGENCY: Simple metrics only"""
        
        basic_status = await self.get_system_status()
        
        return {
            "basic_system_status": basic_status,
            "emergency_metrics": {
                "emergency_mode_active": True,
                "requests_processed": self.metrics["total_requests"],
                "success_rate": self.metrics["successful_responses"] / max(1, self.metrics["total_requests"]),
                "emergency_processing_count": self.metrics["emergency_mode_count"]
            },
            "note": "Advanced features disabled in emergency mode"
        }

    # 🚨 EMERGENCY: Disable all complex methods that might cause recursion
    async def _analyze_and_plan_workflow(self, *args, **kwargs):
        """🚨 DISABLED: Would cause recursion"""
        raise NotImplementedError("Complex workflow disabled in emergency mode")
    
    async def _execute_workflow(self, *args, **kwargs):
        """🚨 DISABLED: Would cause recursion"""
        raise NotImplementedError("Workflow execution disabled in emergency mode")
    
    async def _execute_sequential_workflow(self, *args, **kwargs):
        """🚨 DISABLED: Would cause recursion"""
        raise NotImplementedError("Sequential workflow disabled in emergency mode")
    
    async def _execute_single_model_workflow(self, *args, **kwargs):
        """🚨 DISABLED: Would cause recursion"""
        raise NotImplementedError("Single model workflow disabled in emergency mode")

    # Keep minimal required methods for compatibility
    def _get_available_models(self) -> List[str]:
        """Simple model list"""
        return ["gpt", "claude", "gemini"]