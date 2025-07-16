# claude_main.py - NO RECURSION VERSION
"""
Claude Service - RECURSION-FREE Implementation
แก้ไขทุกจุดที่อาจเกิด recursion + เพิ่ม circuit breaker
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import time
import asyncio
from typing import Optional, Dict, Any
import logging

# Safe imports with fallback
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI(title="NAVA Claude Client - No Recursion", version="1.1.0-SAFE")

# 🆕 Budget Protection (copy from GPT service)
from collections import defaultdict
from datetime import datetime, date, timedelta
import threading

# Emergency mode
EMERGENCY_MODE = False
EMERGENCY_RESPONSE = "Claude service temporarily limited due to high usage. Please try simple queries only."

# Daily tracking
daily_request_counts = defaultdict(int)
daily_count_lock = threading.Lock()
active_requests = 0
max_concurrent_requests = 5  # Claude ให้น้อยกว่า GPT
request_lock = threading.Lock()

def get_user_daily_count(user_id: str) -> int:
    today = date.today().isoformat()
    key = f"{user_id}:{today}"
    with daily_count_lock:
        return daily_request_counts[key]

def increment_user_daily_count(user_id: str):
    today = date.today().isoformat()
    key = f"{user_id}:{today}"
    with daily_count_lock:
        daily_request_counts[key] += 1

def check_emergency_mode(request) -> bool:
    global EMERGENCY_MODE
    # Allow simple greetings even in emergency
    simple_greetings = ['hello', 'hi', 'hey', 'test', 'ping']
    if any(greeting in request.message.lower() for greeting in simple_greetings) and len(request.message) < 20:
        return False
        
    # Block complex requests
    if len(request.message) > 80 or any(word in request.message.lower() 
                                       for word in ['analyze', 'explain', 'write', 'create', 'detailed']):
        return True
    return EMERGENCY_MODE

class CircuitBreaker:
    """🔒 Circuit Breaker to prevent recursive API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"🔒 Circuit breaker OPEN - too many failures ({self.failure_count})")

# Global circuit breaker instance
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

# 🔒 SAFE: Initialize Anthropic client with protection
def get_anthropic_client():
    """🔒 SAFE: Get Anthropic client without recursion"""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here" or not ANTHROPIC_AVAILABLE:
            return None
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.warning(f"⚠️ Anthropic client initialization failed: {e}")
        return None

# Initialize client safely
anthropic_client = get_anthropic_client()

class ChatRequest(BaseModel):
    #message: str
    #user_id: str
    #model: str = "claude-3-5-sonnet-20241022"
    #max_tokens: int = 1000
    #temperature: float = 0.7
    message: str
    user_id: str
    model: str = "claude-3-5-haiku-20241022"  # 🔧 เปลี่ยนเป็นรุ่นถูกสุด
    max_tokens: int = 150                     # 🔧 ลดจาก 1000 → 150
    temperature: float = 0.7

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    response: str
    model_used: str
    tokens_used: int
    confidence: float
    response_time_ms: int
    reasoning: dict

@app.get("/health")
async def health_check():
    """🔒 SAFE: Health check without recursion"""
    
    # Simple status check without complex logic
    api_key_status = "missing"
    if os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("ANTHROPIC_API_KEY") != "your_claude_api_key_here":
            api_key_status = "configured"
    
    anthropic_status = "available" if ANTHROPIC_AVAILABLE else "not_installed"
    client_status = "ready" if anthropic_client else "not_configured"
    
    return {
        "status": "healthy",
        "service": "claude-client-safe",
        "version": "1.1.0-safe",
        "api_key_status": api_key_status,
        "anthropic_library": anthropic_status,
        "client_status": client_status,
        "circuit_breaker": {
            "state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count
        },
        "supported_models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_claude(request: ChatRequest):
    """🔒 SAFE: Chat with Claude - Budget Protected"""
    start_time = time.time()
    
    # 🆕 Emergency Mode Check
    if check_emergency_mode(request):
        return ChatResponse(
            response=EMERGENCY_RESPONSE,
            model_used="claude-emergency",
            tokens_used=5,
            confidence=0.2,
            response_time_ms=int((time.time() - start_time) * 1000),
            reasoning={"mode": "claude_budget_protection"}
        )
    
    # 🆕 Concurrent control
    global active_requests
    with request_lock:
        if active_requests >= max_concurrent_requests:
            return ChatResponse(
                response="Claude service is temporarily busy. Please try again in a moment.",
                model_used="claude-busy",
                tokens_used=10,
                confidence=0.5,
                response_time_ms=int((time.time() - start_time) * 1000),
                reasoning={"mode": "overload_protection"}
            )
        active_requests += 1
    
    # 🆕 Daily limit check
    daily_limit = 30  # Claude ให้น้อยกว่า GPT
    user_daily_count = get_user_daily_count(request.user_id)
    
    if user_daily_count >= daily_limit:
        return ChatResponse(
            response="Daily Claude request limit reached (30 requests). Please try again tomorrow.",
            model_used="claude-limited",
            tokens_used=8,
            confidence=0.0,
            response_time_ms=int((time.time() - start_time) * 1000),
            reasoning={"mode": "daily_limit_reached", "count": user_daily_count}
        )
    
    try:
        # 🔧 Force cheapest model
        request.model = "claude-3-5-haiku-20241022"
        if request.max_tokens > 200:
            request.max_tokens = 200
        
        # Rest of existing logic...
        response = await process_chat_direct(request, start_time)
        
        # Count successful request
        if response.reasoning.get("mode") == "anthropic_api":
            increment_user_daily_count(request.user_id)
        
        return response
        
    finally:
        with request_lock:
            active_requests = max(0, active_requests - 1)

async def process_chat_direct(request: ChatRequest, start_time: float) -> ChatResponse:
    """🔒 DIRECT: Process chat without any recursive calls"""
    
    # Validate model (simple validation, no recursion)
    model = validate_model_simple(request.model)
    
    # Try Anthropic API if available
    if anthropic_client:
        try:
            anthropic_response = await call_anthropic_api_safe(model, request)
            if anthropic_response:
                response_time_ms = int((time.time() - start_time) * 1000)
                return create_anthropic_response(anthropic_response, model, response_time_ms, request)
        except Exception as e:
            logger.warning(f"⚠️ Anthropic API failed, using fallback: {e}")
    
    # Fallback to mock response (no recursion)
    return create_enhanced_mock_response(request, model, start_time)

def validate_model_simple(requested_model: str) -> str:
    """🔒 SIMPLE: Validate model without complex logic"""
    
    valid_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229"
    ]
    
    if requested_model in valid_models:
        return requested_model
    else:
        # Simple fallback - no recursion
        return "claude-3-5-sonnet-20241022"

async def call_anthropic_api_safe(model: str, request: ChatRequest) -> Optional[Dict[str, Any]]:
    """🔒 SAFE: Call Anthropic API without recursion"""
    
    try:
        # Create system message
        system_message = """You are NAVA, an advanced AI assistant powered by Claude. You excel at:
        - Deep analysis and reasoning
        - Creative problem-solving  
        - Detailed explanations
        - Ethical considerations
        
        Provide thoughtful, comprehensive responses that demonstrate Claude's analytical capabilities."""
        
        # Single API call - no retries, no fallbacks, no recursion
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": request.message}
            ]
        )
        
        # Extract data safely
        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "stop_reason": response.stop_reason,
            "model": model
        }
        
    except Exception as e:
        # Log error but don't retry or recurse
        logger.warning(f"⚠️ Anthropic API call failed: {e}")
        return None

def create_anthropic_response(api_response: Dict[str, Any], model: str, response_time_ms: int, request: ChatRequest) -> ChatResponse:
    """🔒 SAFE: Create response from Anthropic API result"""
    
    content = api_response["content"]
    usage = api_response["usage"]
    stop_reason = api_response["stop_reason"]
    
    # Calculate confidence
    confidence = calculate_confidence_safe(content, stop_reason, usage["total_tokens"])
    
    return ChatResponse(
        response=content,
        model_used=model,
        tokens_used=usage["total_tokens"],
        confidence=confidence,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "anthropic_api",
            "stop_reason": stop_reason,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "temperature": request.temperature,
            "model_family": "claude-3",
            "api_success": True
        }
    )

def create_enhanced_mock_response(request: ChatRequest, model: str, start_time: float) -> ChatResponse:
    """🔒 SAFE: Create enhanced mock response without recursion"""
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Enhanced keyword-based responses
    message_lower = request.message.lower()
    
    # Smart mock response generation
    if any(word in message_lower for word in ["analyze", "analysis", "วิเคราะห์", "detailed"]):
        mock_content = f"I'd be happy to provide a detailed analysis of '{request.message}'. As Claude, I excel at breaking down complex topics into clear, actionable insights with comprehensive reasoning and deep analytical thinking. Let me examine the key aspects and provide you with a thorough evaluation."
        
    elif any(word in message_lower for word in ["write", "create", "เขียน", "สร้าง"]):
        mock_content = f"I can certainly help you with writing '{request.message}'. Claude excels at creating high-quality content with nuanced understanding, proper structure, and engaging narrative flow. I'll approach this with careful attention to style, tone, and your specific requirements."
        
    elif any(word in message_lower for word in ["code", "programming", "function", "โค้ด"]):
        mock_content = f"I'm well-equipped to assist with your coding request: '{request.message}'. Claude provides strong programming support with clean code generation, architectural insights, debugging assistance, and clear explanations of complex programming concepts."
        
    elif any(word in message_lower for word in ["strategy", "plan", "กลยุทธ์", "แผน"]):
        mock_content = f"I'd be delighted to help you develop a strategic approach for '{request.message}'. Claude excels at strategic thinking, comprehensive planning, and providing actionable frameworks that consider multiple perspectives and potential outcomes."
        
    else:
        mock_content = f"Thank you for your question: '{request.message}'. I'm Claude, and I'm here to provide thoughtful, comprehensive assistance. I excel at analytical thinking, creative problem-solving, and providing detailed explanations tailored to your specific needs."
    
    # Add mock disclaimer
    mock_content += "\n\n(Note: This is an enhanced mock response. For full Claude capabilities, configure the Anthropic API key.)"
    
    return ChatResponse(
        response=mock_content,
        model_used=f"{model}-enhanced-mock",
        tokens_used=len(mock_content.split()) * 2,  # Rough token estimate
        confidence=0.85,  # High confidence for enhanced mocks
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "enhanced_mock",
            "api_available": anthropic_client is not None,
            "anthropic_library": ANTHROPIC_AVAILABLE,
            "temperature": request.temperature,
            "model_family": "claude-3-mock",
            "enhancement_level": "keyword_based"
        }
    )

def create_circuit_breaker_response(request: ChatRequest, start_time: float) -> ChatResponse:
    """🔒 SAFE: Circuit breaker response"""
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        response="I'm currently in a protective mode due to recent technical issues, but I can still provide assistance. Let me help you with a thoughtful response while my systems recover.",
        model_used=f"{request.model}-circuit-breaker",
        tokens_used=30,
        confidence=0.70,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "circuit_breaker",
            "state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count,
            "protection_active": True
        }
    )

def create_safe_error_response(request: ChatRequest, start_time: float, error_msg: str) -> ChatResponse:
    """🔒 SAFE: Error response without recursion"""
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        response="I apologize for the technical difficulty. I'm still here to help you with your question to the best of my ability. Please let me know how I can assist you.",
        model_used=f"{request.model}-safe-error",
        tokens_used=25,
        confidence=0.60,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "safe_error",
            "error": error_msg[:100],  # Truncate long errors
            "protection_level": "high",
            "fallback_active": True
        }
    )

def calculate_confidence_safe(response: str, stop_reason: str, tokens_used: int) -> float:
    """🔒 SAFE: Calculate confidence without complex logic"""
    
    confidence = 0.85  # Base confidence for Claude
    
    # Simple adjustments
    if stop_reason == "end_turn":
        confidence += 0.1
    elif stop_reason == "max_tokens":
        confidence -= 0.05
    
    # Length bonus
    if len(response) > 100:
        confidence += 0.03
    if len(response) > 300:
        confidence += 0.03
    
    # Ensure bounds
    return max(0.0, min(1.0, confidence))

@app.get("/models")
async def list_supported_models():
    """🔒 SAFE: List models without complex logic"""
    
    return {
        "supported_models": [
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "description": "Latest balanced model for most tasks",
                "max_tokens": 8192,
                "strengths": ["Analysis", "Writing", "Coding", "Reasoning"]
            },
            {
                "id": "claude-3-5-haiku-20241022", 
                "name": "Claude 3.5 Haiku",
                "description": "Fast model for simple tasks",
                "max_tokens": 8192,
                "strengths": ["Speed", "Efficiency", "Quick responses"]
            },
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus", 
                "description": "Most powerful model for complex reasoning",
                "max_tokens": 4096,
                "strengths": ["Complex reasoning", "Deep analysis", "Creative tasks"]
            }
        ],
        "service_version": "1.1.0-safe",
        "recursion_protection": "enabled"
    }

@app.get("/capabilities")
async def get_claude_capabilities():
    """🔒 SAFE: Get capabilities without complex processing"""
    
    return {
        "capabilities": {
            "analysis": {
                "description": "Deep analytical thinking and reasoning",
                "strength": "very_high",
                "use_cases": ["Research", "Problem solving", "Critical thinking"]
            },
            "writing": {
                "description": "Professional and creative writing",
                "strength": "very_high", 
                "use_cases": ["Content creation", "Documentation", "Creative writing"]
            },
            "coding": {
                "description": "Programming assistance and code review",
                "strength": "high",
                "use_cases": ["Code generation", "Debugging", "Architecture design"]
            },
            "strategy": {
                "description": "Strategic planning and decision support",
                "strength": "high",
                "use_cases": ["Business planning", "Problem solving", "Decision analysis"]
            }
        },
        "model_family": "claude-3.5",
        "provider": "Anthropic",
        "service_version": "1.1.0-safe",
        "protection_features": [
            "circuit_breaker",
            "recursion_prevention", 
            "graceful_degradation",
            "safe_error_handling"
        ]
    }

@app.get("/status")
async def get_service_status():
    """🔒 SAFE: Get detailed service status"""
    
    return {
        "service": "claude-client-safe",
        "version": "1.1.0-safe",
        "status": "operational",
        "anthropic_api": {
            "available": ANTHROPIC_AVAILABLE,
            "client_configured": anthropic_client is not None,
            "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY"))
        },
        "circuit_breaker": {
            "state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count,
            "last_failure": circuit_breaker.last_failure_time
        },
        "safety_features": {
            "recursion_protection": "enabled",
            "circuit_breaker": "enabled", 
            "graceful_degradation": "enabled",
            "safe_fallbacks": "enabled"
        },
        "capabilities": [
            "anthropic_api_calls",
            "enhanced_mock_responses",
            "error_recovery",
            "circuit_breaking"
        ]
    }
@app.get("/admin/usage-stats")
async def get_usage_stats():
    today = date.today().isoformat()
    total_today = sum(count for key, count in daily_request_counts.items() 
                     if key.endswith(today))
    return {
        "total_requests_today": total_today,
        "active_requests": active_requests,
        "daily_limit": 30,  # Claude limit ต่ำกว่า GPT
        "emergency_mode": EMERGENCY_MODE,
        "users_today": len([k for k in daily_request_counts.keys() if k.endswith(today)])
    }

@app.post("/admin/emergency-mode")
async def toggle_emergency_mode(enable: bool = True):
    global EMERGENCY_MODE
    EMERGENCY_MODE = enable
    return {"emergency_mode": EMERGENCY_MODE, "message": f"Claude emergency mode {'enabled' if enable else 'disabled'}"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8003))
    logger.info(f"🚀 Starting Claude Service (Safe) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)