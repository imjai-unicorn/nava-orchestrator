from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
import time
from typing import Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="NAVA GPT Client", version="1.0.0")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    user_id: str
    model: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: int
    confidence: float
    response_time_ms: int
    reasoning: dict

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    
    return {
        "status": "healthy",
        "service": "gpt-client",
        "version": "1.0.0",
        "api_key_status": api_key_status,
        "supported_models": ["gpt-4", "gpt-3.5-turbo"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gpt(request: ChatRequest):
    """Chat with OpenAI GPT models"""
    start_time = time.time()
    
    try:
        # Check if API key is configured
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            # Return mock response for testing
            return _create_mock_response(request, start_time)
        
        # Validate model
        valid_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
        if request.model not in valid_models:
            request.model = "gpt-4"
        
        # Create system message for NAVA context
        system_message = """You are NAVA, an advanced AI assistant. Provide helpful, accurate, and engaging responses. 
        Consider the user's context and provide actionable insights when appropriate."""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.message}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            user=request.user_id
        )
        
        # Extract response data
        ai_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        finish_reason = response.choices[0].finish_reason
        
        # Calculate confidence based on response quality
        confidence = _calculate_confidence(ai_response, finish_reason, tokens_used)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
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