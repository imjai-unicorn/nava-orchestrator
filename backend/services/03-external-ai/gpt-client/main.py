from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
import time
from typing import Optional
import asyncio
from app.timeout_handler import gpt_timeout_handler, call_gpt_with_timeout

# Load environment variables
load_dotenv()

app = FastAPI(title="NAVA GPT Client", version="1.0.0")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🆕 Concurrent request management
import threading
active_requests = 0
max_concurrent_requests = 8  # จำกัดไม่เกิน 8 concurrent requests
request_lock = threading.Lock()
request_queue = asyncio.Queue(maxsize=20)

from collections import defaultdict
from datetime import datetime, date, timedelta

daily_request_counts = defaultdict(int)
daily_count_lock = threading.Lock()

def get_user_daily_count(user_id: str) -> int:
    """Get user's daily request count"""
    today = date.today().isoformat()
    key = f"{user_id}:{today}"
    
    with daily_count_lock:
        return daily_request_counts[key]

def increment_user_daily_count(user_id: str):
    """Increment user's daily request count"""
    today = date.today().isoformat()
    key = f"{user_id}:{today}"
    
    with daily_count_lock:
        daily_request_counts[key] += 1
        
        # Clean old dates (keep only today and yesterday)
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        keys_to_remove = [k for k in daily_request_counts.keys() 
                         if not (k.endswith(today) or k.endswith(yesterday))]
        for k in keys_to_remove:
            del daily_request_counts[k]

class ChatRequest(BaseModel):
    #message: str
    #user_id: str
    #model: str = "gpt-4"
    #max_tokens: int = 1000
    #temperature: float = 0.7
    message: str
    user_id: str
    model: str = "gpt-3.5-turbo"  # 🔧 เปลี่ยนเป็นรุ่นถูก
    max_tokens: int = 150         # 🔧 ลดจาก 1000 → 150
    temperature: float = 0.7

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # ใช้แค่อันนี้
    response: str
    model_used: str
    tokens_used: int
    confidence: float
    response_time_ms: int
    reasoning: dict

# 🆕 Emergency budget protection (moved after class definitions)

EMERGENCY_MODE = False  # Set to True if budget running low
EMERGENCY_RESPONSE = "Service temporarily limited due to high usage. Please try simple queries only."

def check_emergency_mode(request: ChatRequest) -> bool:
    """Check if we should enter emergency mode"""
    global EMERGENCY_MODE
    
    # Enable emergency mode for complex requests
    if len(request.message) > 100 or any(word in request.message.lower() 
                                        for word in ['analyze', 'explain', 'write', 'create', 'generate']):
        return True
    
    return EMERGENCY_MODE
        
@app.get("/health")
async def health_check():
    """Health check endpoint with timeout monitoring"""
    api_key_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    
    # 🆕 Get timeout handler health
    timeout_health = gpt_timeout_handler.get_health_status()
    
    return {
        "status": "healthy" if not timeout_health["circuit_breaker_open"] else "degraded",
        "service": "gpt-client", 
        "version": "1.0.0",
        "api_key_status": api_key_status,
        "supported_models": ["gpt-4", "gpt-3.5-turbo"],
        "timeout_handler": timeout_health
    }

@app.get("/health/timeout")
async def timeout_health():
    """Detailed timeout handler health"""
    return gpt_timeout_handler.get_health_status()

@app.post("/admin/reset-circuit-breaker")
async def reset_circuit_breaker():
    """Admin endpoint to reset circuit breaker"""
    gpt_timeout_handler.reset_circuit_breaker()
    return {"message": "Circuit breaker reset successfully"}

@app.get("/admin/usage-stats")
async def get_usage_stats():
    """Get current usage statistics"""
    today = date.today().isoformat()
    
    total_today = sum(count for key, count in daily_request_counts.items() 
                     if key.endswith(today))
    
    return {
        "total_requests_today": total_today,
        "active_requests": active_requests,
        "daily_limit": 50,
        "emergency_mode": EMERGENCY_MODE,
        "users_today": len([k for k in daily_request_counts.keys() if k.endswith(today)])
    }

@app.post("/admin/emergency-mode")
async def toggle_emergency_mode(enable: bool = True):
    """Toggle emergency mode"""
    global EMERGENCY_MODE
    EMERGENCY_MODE = enable
    return {"emergency_mode": EMERGENCY_MODE, "message": f"Emergency mode {'enabled' if enable else 'disabled'}"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gpt(request: ChatRequest):
    """Chat with OpenAI GPT models with timeout handling"""
    start_time = time.time()
    
    # 🆕 Determine complexity from message length and content
    complexity = "simple"
    if len(request.message) > 100:
        complexity = "medium"
    if len(request.message) > 300 or "analyze" in request.message.lower():
        complexity = "complex"
    if "strategy" in request.message.lower() or "plan" in request.message.lower():
        complexity = "critical"
    
    # 🆕 Emergency Mode Check
    if check_emergency_mode(request):
        return ChatResponse(
            response=EMERGENCY_RESPONSE,
            model_used="emergency-mode",
            tokens_used=5,
            confidence=0.2,
            response_time_ms=int((time.time() - start_time) * 1000),
            reasoning={"mode": "emergency_budget_protection"}
        )
    
    
    # 🆕 Concurrent control
    global active_requests
    with request_lock:
        if active_requests >= max_concurrent_requests:
            # Return immediate response for overload
            return ChatResponse(
                response="Service is temporarily busy. Please try again in a moment.",
                model_used=f"{request.model}-busy",
                tokens_used=15,
                confidence=0.5,
                response_time_ms=int((time.time() - start_time) * 1000),
                reasoning={"mode": "overload_protection", "active_requests": active_requests}
            )
        
        active_requests += 1
    
    # 🆕 Cost Control - จำกัดการใช้งานรายวัน
    daily_request_limit = 50  # จำกัด 50 requests ต่อวัน
    user_daily_count = get_user_daily_count(request.user_id)
    
    if user_daily_count >= daily_request_limit:
        return ChatResponse(
            response="Daily request limit reached (50 requests). Please try again tomorrow.",
            model_used=f"{request.model}-limited",
            tokens_used=10,
            confidence=0.0,
            response_time_ms=int((time.time() - start_time) * 1000),
            reasoning={"mode": "daily_limit_reached", "count": user_daily_count}
        )
           
    try:
        # Check if API key is configured
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            # Return mock response for testing
            return _create_mock_response(request, start_time)
        
        # 🔧 Force ใช้ GPT-3.5 turbo เพื่อประหยัดค่าใช้จ่าย
        request.model = "gpt-3.5-turbo"
        
        # 🔧 จำกัด max_tokens สำหรับ testing
        if request.max_tokens > 200:
            request.max_tokens = 200  # ลดค่าใช้จ่าย

        # 🔧 ใช้ system message แบบสั้น
        system_message = "You are NAVA AI. Give concise, helpful responses."
        
        # 🆕 Call OpenAI API with timeout handling
        async def make_openai_call():
            return client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.message}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                user=request.user_id
            )
        
        # Use timeout handler
        timeout_result = await call_gpt_with_timeout(
            make_openai_call,
            complexity=complexity,
            fallback=lambda: _create_timeout_fallback_response(request)
        )
        
        if not timeout_result.get("success", False):
            # Handle timeout/error
            return _create_error_fallback_response(request, start_time, timeout_result.get("message", "Unknown error"))
        response = timeout_result["response"]
        
        # Extract response data
        ai_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        finish_reason = response.choices[0].finish_reason
        
        # Calculate confidence based on response quality
        confidence = _calculate_confidence(ai_response, finish_reason, tokens_used)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # 🆕 Count successful request
        increment_user_daily_count(request.user_id)

        return ChatResponse(
            response=ai_response,
            model_used=request.model,
            tokens_used=tokens_used,
            confidence=confidence,
            response_time_ms=response_time_ms,
            reasoning={
                "finish_reason": finish_reason,
                "temperature": request.temperature,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        )
        
    except Exception as e:
        # Log error and return fallback response
        error_msg = str(e)
        print(f"OpenAI API Error: {error_msg}")
        
        # Return graceful fallback
        return _create_error_fallback_response(request, start_time, error_msg)
        
    finally:
        # 🆕 Cleanup concurrent count
        with request_lock:
            active_requests = max(0, active_requests - 1)
       
def _create_mock_response(request: ChatRequest, start_time: float) -> ChatResponse:
    """Create mock response when API key is not configured"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    mock_responses = {
        "hello": "Hello! I'm NAVA, your AI assistant. How can I help you today?",
        "code": "I can help you with coding! What programming language or problem are you working on?",
        "analysis": "I'd be happy to help analyze that for you. Could you provide more details?",
        "default": f"Thank you for your message: '{request.message}'. I'm ready to assist you!"
    }
    
    # Simple keyword matching for mock responses
    message_lower = request.message.lower()
    if "hello" in message_lower or "hi" in message_lower:
        response_text = mock_responses["hello"]
    elif "code" in message_lower or "programming" in message_lower:
        response_text = mock_responses["code"]
    elif "analyze" in message_lower or "analysis" in message_lower:
        response_text = mock_responses["analysis"]
    else:
        response_text = mock_responses["default"]
    
    return ChatResponse(
        response=response_text + " (Note: This is a mock response - configure OpenAI API key for real responses)",
        model_used=f"{request.model}-mock",
        tokens_used=len(response_text.split()) * 2,  # Rough token estimate
        confidence=0.85,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "mock",
            "api_key_status": "not_configured",
            "temperature": request.temperature
        }
    )

def _create_error_fallback_response(request: ChatRequest, start_time: float, error_msg: str) -> ChatResponse:
    """Create fallback response when API call fails"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        response="I apologize, but I'm experiencing technical difficulties right now. Please try again in a moment.",
        model_used=f"{request.model}-error",
        tokens_used=20,
        confidence=0.1,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "error_fallback",
            "error": error_msg[:100],  # Truncate long errors
            "temperature": request.temperature
        }
    )

def _calculate_confidence(response: str, finish_reason: str, tokens_used: int) -> float:
    """Calculate confidence score based on response quality indicators"""
    confidence = 0.8  # Base confidence
    
    # Adjust based on finish reason
    if finish_reason == "stop":
        confidence += 0.1
    elif finish_reason == "length":
        confidence -= 0.1
    
    # Adjust based on response length
    if len(response) > 50:
        confidence += 0.05
    if len(response) > 200:
        confidence += 0.05
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))
def _create_timeout_fallback_response(request: ChatRequest):
    """Create fallback response for timeout situations"""
    return {
        "choices": [{
            "message": {
                "content": "I apologize for the delay. Let me provide a quick response to your query. Could you please rephrase or simplify your question?"
            },
            "finish_reason": "timeout_fallback"
        }],
        "usage": {
            "total_tokens": 25,
            "prompt_tokens": 15,
            "completion_tokens": 10
        }
    }
@app.get("/models")
async def list_supported_models():
    """List supported OpenAI models"""
    return {
        "supported_models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "description": "Most capable model, best for complex tasks",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.03
            },
            {
                "id": "gpt-3.5-turbo", 
                "name": "GPT-3.5 Turbo",
                "description": "Fast and efficient for most tasks",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.002
            },
            {
                "id": "gpt-4-turbo-preview",
                "name": "GPT-4 Turbo",
                "description": "Latest GPT-4 with improved performance",
                "max_tokens": 128000,
                "cost_per_1k_tokens": 0.01
            }
        ]
    }

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)