from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import google.generativeai as genai
import time
from typing import Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="NAVA Gemini Client", version="1.0.0")

# Configure Google AI
api_key = os.getenv("GOOGLE_AI_API_KEY")
if api_key and api_key != "your_gemini_api_key_here":
    genai.configure(api_key=api_key)

class ChatRequest(BaseModel):
    message: str
    user_id: str
    model: str = "gemini-2.0-flash-exp"
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
    api_key_status = "configured" if (os.getenv("GOOGLE_AI_API_KEY") and 
                                     os.getenv("GOOGLE_AI_API_KEY") != "your_gemini_api_key_here") else "missing"
    
    return {
        "status": "healthy",
        "service": "gemini-client",
        "version": "1.0.0",
        "api_key_status": api_key_status,
        "supported_models": ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    """Chat with Google Gemini models"""
    start_time = time.time()
    
    try:
        # Check if API key is configured
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            # Return mock response for testing
            return _create_mock_response(request, start_time)
        
        # Validate model
        valid_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
        if request.model not in valid_models:
            request.model = "gemini-2.0-flash-exp"
        
        # Create model instance
        model = genai.GenerativeModel(request.model)
        
        # Create enhanced prompt for NAVA context
        system_context = """You are NAVA, an advanced AI assistant powered by Google's Gemini. You excel at:
        - Multimodal understanding and analysis
        - Creative problem-solving with visual thinking
        - Real-time information processing
        - Technical explanations with visual examples
        - Innovative solutions and fresh perspectives
        
        Provide comprehensive, innovative responses that showcase Gemini's advanced capabilities."""
        
        full_prompt = f"{system_context}\n\nUser: {request.message}"
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.8,
            top_k=40
        )
        
        # Call Gemini API
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Extract response data
        ai_response = response.text
        
        # Estimate tokens (Gemini doesn't provide exact counts)
        tokens_used = _estimate_tokens(full_prompt, ai_response)
        
        # Get additional info if available
        finish_reason = getattr(response.candidates[0], 'finish_reason', 'STOP') if response.candidates else 'STOP'
        safety_ratings = getattr(response.candidates[0], 'safety_ratings', []) if response.candidates else []
        
        # Calculate confidence based on response quality
        confidence = _calculate_confidence(ai_response, finish_reason, safety_ratings)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            response=ai_response,
            model_used=request.model,
            tokens_used=tokens_used,
            confidence=confidence,
            response_time_ms=response_time_ms,
            reasoning={
                "finish_reason": str(finish_reason),
                "temperature": request.temperature,
                "safety_ratings": [{"category": str(rating.category), "probability": str(rating.probability)} 
                                 for rating in safety_ratings],
                "model_family": "gemini",
                "estimated_tokens": True
            }
        )
        
    except Exception as e:
        # Log error and return fallback response
        error_msg = str(e)
        print(f"Google AI API Error: {error_msg}")
        
        # Return graceful fallback
        return _create_error_fallback_response(request, start_time, error_msg)

def _create_mock_response(request: ChatRequest, start_time: float) -> ChatResponse:
    """Create mock response when API key is not configured"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    mock_responses = {
        "multimodal": "I excel at multimodal understanding! I can process text, images, and even help visualize concepts. What would you like to explore?",
        "creative": "Let me think creatively about this! As Gemini, I love exploring innovative approaches and generating fresh perspectives on complex problems.",
        "technical": "I can provide detailed technical explanations with visual examples and step-by-step breakdowns. What technical concept would you like me to explain?",
        "realtime": "I'm designed to work with real-time information and provide up-to-date insights. How can I assist you with current information?",
        "visual": "I can help you think visually about problems and create conceptual frameworks. What would you like to visualize or map out?",
        "default": f"Hello! I'm NAVA powered by Gemini. Your message: '{request.message}' - I'm ready to provide innovative, multimodal assistance!"
    }
    
    # Enhanced keyword matching for mock responses
    message_lower = request.message.lower()
    if any(word in message_lower for word in ["image", "visual", "picture", "diagram", "chart"]):
        response_text = mock_responses["visual"]
    elif any(word in message_lower for word in ["creative", "innovative", "idea", "brainstorm"]):
        response_text = mock_responses["creative"]
    elif any(word in message_lower for word in ["technical", "explain", "how", "programming", "code"]):
        response_text = mock_responses["technical"]
    elif any(word in message_lower for word in ["current", "latest", "recent", "update", "news"]):
        response_text = mock_responses["realtime"]
    elif any(word in message_lower for word in ["multimodal", "different", "types", "formats"]):
        response_text = mock_responses["multimodal"]
    else:
        response_text = mock_responses["default"]
    
    return ChatResponse(
        response=response_text + " (Note: This is a mock response - configure Google AI API key for real Gemini responses)",
        model_used=f"{request.model}-mock",
        tokens_used=len(response_text.split()) * 2,  # Rough token estimate
        confidence=0.90,  # Gemini typically very confident
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "mock",
            "api_key_status": "not_configured",
            "temperature": request.temperature,
            "model_family": "gemini-mock",
            "estimated_tokens": True
        }
    )

def _create_error_fallback_response(request: ChatRequest, start_time: float, error_msg: str) -> ChatResponse:
    """Create fallback response when API call fails"""
    response_time_ms = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        response="I'm experiencing some technical difficulties right now. Please try again in a moment, and I'll provide you with innovative assistance!",
        model_used=f"{request.model}-error",
        tokens_used=30,
        confidence=0.1,
        response_time_ms=response_time_ms,
        reasoning={
            "mode": "error_fallback",
            "error": error_msg[:100],  # Truncate long errors
            "temperature": request.temperature,
            "model_family": "gemini-error",
            "estimated_tokens": True
        }
    )

def _estimate_tokens(prompt: str, response: str) -> int:
    """Estimate token count (Gemini doesn't provide exact counts)"""
    # Rough estimation: 1 token ≈ 0.75 words
    total_words = len(prompt.split()) + len(response.split())
    return int(total_words * 1.33)  # Convert words to approximate tokens

def _calculate_confidence(response: str, finish_reason: str, safety_ratings: list) -> float:
    """Calculate confidence score based on Gemini response quality indicators"""
    confidence = 0.88  # Base confidence (Gemini typically high confidence)
    
    # Adjust based on finish reason
    finish_reason_str = str(finish_reason)
    if "STOP" in finish_reason_str:
        confidence += 0.08
    elif "MAX_TOKENS" in finish_reason_str:
        confidence -= 0.05
    
    # Adjust based on safety ratings
    if safety_ratings:
        high_safety_count = sum(1 for rating in safety_ratings 
                              if "LOW" in str(rating.get('probability', '')))
        if high_safety_count == len(safety_ratings):
            confidence += 0.04
    
    # Adjust based on response quality
    if len(response) > 100:
        confidence += 0.05
    if len(response) > 300:
        confidence += 0.03
    
    # Gemini bonus for innovative/detailed responses
    if any(word in response.lower() for word in ["innovative", "creative", "perspective", "approach"]):
        confidence += 0.02
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

@app.get("/models")
async def list_supported_models():
    """List supported Google Gemini models"""
    return {
        "supported_models": [
            {
                "id": "gemini-2.0-flash-exp",
                "name": "Gemini 2.0 Flash (Experimental)",
                "description": "Latest and most advanced Gemini model",
                "max_tokens": 8192,
                "context_window": 1000000,
                "strengths": ["Latest capabilities", "Enhanced reasoning", "Multimodal++"]
            },
            {
                "id": "gemini-2.0-flash-thinking-exp",
                "name": "Gemini 2.0 Flash Thinking (Experimental)",
                "description": "Advanced reasoning model like GPT-o1",
                "max_tokens": 8192,
                "context_window": 1000000,
                "strengths": ["Deep reasoning", "Step-by-step thinking", "Complex problems"]
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "description": "Stable and capable model with 1M token context",
                "max_tokens": 8192,
                "context_window": 1000000,
                "strengths": ["Long context", "Complex reasoning", "Multimodal"]
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash", 
                "description": "Fast model optimized for speed",
                "max_tokens": 8192,
                "context_window": 1000000,
                "strengths": ["Speed", "Efficiency", "Quick responses"]
            }
        ]
    }

@app.get("/capabilities")
async def get_gemini_capabilities():
    """Get Gemini-specific capabilities"""
    return {
        "capabilities": {
            "multimodal": {
                "description": "Text, image, and video understanding",
                "strength": "very_high",
                "use_cases": ["Image analysis", "Visual Q&A", "Content creation"],
                "supported_formats": ["text", "images", "video"]
            },
            "long_context": {
                "description": "1M token context window",
                "strength": "very_high",
                "use_cases": ["Document analysis", "Long conversations", "Research"]
            },
            "realtime": {
                "description": "Real-time information processing",
                "strength": "high",
                "use_cases": ["Current events", "Live analysis", "Updates"]
            },
            "creative": {
                "description": "Creative and innovative thinking",
                "strength": "very_high", 
                "use_cases": ["Content creation", "Brainstorming", "Innovation"]
            },
            "coding": {
                "description": "Programming and technical assistance",
                "strength": "high",
                "use_cases": ["Code generation", "Debugging", "Technical docs"]
            }
        },
        "model_family": "gemini",
        "provider": "Google AI",
        "unique_features": [
            "1M token context window",
            "Multimodal understanding",
            "Real-time capabilities",
            "Advanced safety features"
        ]
    }

@app.get("/safety")
async def get_safety_info():
    """Get Gemini safety and content filtering information"""
    return {
        "safety_features": {
            "content_filtering": {
                "description": "Advanced content safety filtering",
                "categories": ["harassment", "hate_speech", "sexually_explicit", "dangerous_content"],
                "adjustable": True
            },
            "harm_prevention": {
                "description": "Proactive harm prevention",
                "enabled": True,
                "real_time": True
            }
        },
        "safety_ratings": {
            "explanation": "Each response includes safety ratings",
            "scale": "NEGLIGIBLE, LOW, MEDIUM, HIGH",
            "categories": ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                          "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)