from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import anthropic
import time
from typing import Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="NAVA Claude Client", version="1.0.0")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    user_id: str
    model: str = "claude-3-5-sonnet-20241022"
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
    api_key_status = "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing"
    
    return {
        "status": "healthy",
        "service": "claude-client",
        "version": "1.0.0",
        "api_key_status": api_key_status,
        "supported_models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_claude(request: ChatRequest):
    """Chat with Anthropic Claude models"""
    start_time = time.time()
    
    try:
        # Check if API key is configured
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here":
            # Return mock response for testing
            return _create_mock_response(request, start_time)
        
        # Validate model
        valid_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240229"
        ]
        if request.model not in valid_models:
            request.model = "claude-3-5-sonnet-20241022"
        
        # Create system message for NAVA context
        system_message = """You are NAVA, an advanced AI assistant powered by Claude. You excel at:
        - Deep analysis and reasoning
        - Creative problem-solving
        - Detailed explanations
        - Ethical considerations
        
        Provide thoughtful, comprehensive responses that demonstrate Claude's analytical capabilities."""
        
        # Call Anthropic API
        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": request.message}
            ]
        )
        
        # Extract response data
        ai_response = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        stop_reason = response.stop_reason
        
        # Calculate confidence based on response quality
        confidence = _calculate_confidence(ai_response, stop_reason, tokens_used)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            response=ai_response,
            model_used=request.model,
            tokens_used=tokens_used,
            confidence=confidence,
            response_time_ms=response_time_ms,
            reasoning={
                "stop_reason": stop_reason,
                "temperature": request.temperature,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model_family": "claude-3"
            }
        )
        
    except Exception as e:
        # Log error and return fallback response
        error_msg = str(e)
        print(f"Anthropic API Error: {error_msg}")
        
        # Return graceful fallback
        return _create_error_fallback_response(request, start_time, error_msg)

def _create_mock_response(request: ChatRequest, start_time: float) -> ChatResponse:
    """Create mock response when API key is not configured"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    mock_responses = {
        "analysis": "I'd be delighted to provide a thorough analysis of that topic. Claude excels at breaking down complex subjects into clear, actionable insights.",
        "creative": "I can help with creative tasks! As Claude, I enjoy exploring innovative solutions and thinking outside conventional boundaries.",
        "code": "I'm well-equipped to assist with coding challenges. I can help with code review, debugging, architecture design, and explaining complex programming concepts.",
        "ethical": "That's an important ethical consideration. Let me think through the various perspectives and implications of this situation.",
        "default": f"Thank you for your thoughtful question: '{request.message}'. I'm Claude, and I'm here to provide comprehensive, nuanced assistance."
    }
    
    # Enhanced keyword matching for mock responses
    message_lower = request.message.lower()
    if any(word in message_lower for word in ["analyze", "analysis", "breakdown", "explain"]):
        response_text = mock_responses["analysis"]
    elif any(word in message_lower for word in ["creative", "design", "innovative", "idea"]):
        response_text = mock_responses["creative"]
    elif any(word in message_lower for word in ["code", "programming", "debug", "function"]):
        response_text = mock_responses["code"]
    elif any(word in message_lower for word in ["ethical", "moral", "right", "wrong", "should"]):
        response_text = mock_responses["ethical"]
    else:
        response_text = mock_responses["default"]
    
    return ChatResponse(
        response=response_text + " (Note: This is a mock response - configure Anthropic API key for real Claude responses)",
        model_used=f"{request.model}-mock",
        tokens_used=len(response_text.split()) * 2,  # Rough token estimate
        confidence=0.88,  # Claude typically has high confidence
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "mock",
            "api_key_status": "not_configured",
            "temperature": request.temperature,
            "model_family": "claude-3-mock"
        }
    )

def _create_error_fallback_response(request: ChatRequest, start_time: float, error_msg: str) -> ChatResponse:
    """Create fallback response when API call fails"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        response="I apologize, but I'm experiencing some technical difficulties at the moment. Please try again shortly, and I'll do my best to assist you.",
        model_used=f"{request.model}-error",
        tokens_used=25,
        confidence=0.1,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "error_fallback",
            "error": error_msg[:100],  # Truncate long errors
            "temperature": request.temperature,
            "model_family": "claude-3-error"
        }
    )

def _calculate_confidence(response: str, stop_reason: str, tokens_used: int) -> float:
    """Calculate confidence score based on Claude response quality indicators"""
    confidence = 0.85  # Base confidence (Claude typically more confident)
    
    # Adjust based on stop reason
    if stop_reason == "end_turn":
        confidence += 0.1
    elif stop_reason == "max_tokens":
        confidence -= 0.05
    
    # Adjust based on response length and depth
    if len(response) > 100:
        confidence += 0.05
    if len(response) > 300:
        confidence += 0.05
    
    # Claude bonus for detailed responses
    if "analysis" in response.lower() or "consider" in response.lower():
        confidence += 0.02
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

@app.get("/models")
async def list_supported_models():
    """List supported Anthropic Claude models"""
    return {
        "supported_models": [
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "description": "Most powerful model for complex reasoning and analysis",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.015,
                "strengths": ["Complex reasoning", "Creative writing", "Detailed analysis"]
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3-5 Sonnet", 
                "description": "Balanced model for most tasks",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.003,
                "strengths": ["General tasks", "Coding", "Analysis"]
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3-5 Haiku",
                "description": "Fastest model for simple tasks",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.00025,
                "strengths": ["Quick responses", "Simple tasks", "Cost efficiency"]
            }
        ]
    }

@app.get("/capabilities")
async def get_claude_capabilities():
    """Get Claude-specific capabilities"""
    return {
        "capabilities": {
            "analysis": {
                "description": "Deep analytical thinking and reasoning",
                "strength": "very_high",
                "use_cases": ["Research", "Problem solving", "Critical thinking"]
            },
            "coding": {
                "description": "Programming assistance and code review",
                "strength": "high", 
                "use_cases": ["Code generation", "Debugging", "Architecture design"]
            },
            "creativity": {
                "description": "Creative writing and ideation",
                "strength": "very_high",
                "use_cases": ["Content creation", "Brainstorming", "Creative problem solving"]
            },
            "ethics": {
                "description": "Ethical reasoning and considerations",
                "strength": "very_high",
                "use_cases": ["Ethical analysis", "Decision making", "Policy review"]
            }
        },
        "model_family": "claude-3",
        "provider": "Anthropic"
    }

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)