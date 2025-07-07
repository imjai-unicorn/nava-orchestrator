# main.py - Fixed JSON Response
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

# Import ที่แก้ให้ตรงกับไฟล์จริง
from app.service.logic_orchestrator import LogicOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="NAVA Logic Controller",
    description="Simple AI Orchestration System",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models - Fixed to match response
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = Field("anonymous")
    user_preference: Optional[str] = Field(None)
    context: Optional[Dict[str, Any]] = Field(None)

class ChatResponse(BaseModel):
    response: str
    model_used: str
    confidence: float
    processing_time_seconds: float = 0.0
    orchestration_type: str = "logic_system"
    workflow_used: bool = False
    decision_info: Dict[str, Any] = {}
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    nava_initialized: bool
    ai_services: Optional[Dict[str, Any]] = None
    available_models: List[str] = []
    uptime_seconds: float
    timestamp: str

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    logger.info("Starting NAVA Logic Controller...")
    try:
        orchestrator = LogicOrchestrator()
        await orchestrator.initialize()
        logger.info("NAVA Logic Controller started successfully")
    except Exception as e:
        logger.error(f"Failed to start NAVA: {e}")
        # Don't raise - let it continue with fallback
        orchestrator = LogicOrchestrator()

# Main chat endpoint - Fixed
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    
    try:
        logger.info(f"Chat request from user: {request.user_id}")
        
        # Process through orchestrator
        if orchestrator and orchestrator.is_initialized:
            result = await orchestrator.process_request(
                message=request.message,
                user_id=request.user_id,
                user_preference=request.user_preference,
                context=request.context
            )
        else:
            # Fallback response
            result = {
                "response": f"Simple NAVA response: {request.message}",
                "model_used": "nava_fallback",
                "confidence": 0.8,
                "processing_time_seconds": 0.1
            }

        # Ensure all required fields with proper types
        response_data = {
            "response": str(result.get("response", "No response")),
            "model_used": str(result.get("model_used", "nava_fallback")),
            "confidence": float(result.get("confidence", 0.8)),
            "processing_time_seconds": float(result.get("processing_time_seconds", 0.1)),
            "orchestration_type": str(result.get("orchestration_type", "logic_system")),
            "workflow_used": bool(result.get("workflow_used", False)),
            "decision_info": dict(result.get("decision_info", {})),
            "timestamp": str(result.get("timestamp", datetime.now().isoformat()))
        }

        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Return proper error response
        return ChatResponse(
            response=f"Error processing request: {str(e)}",
            model_used="error",
            confidence=0.0,
            processing_time_seconds=0.0,
            orchestration_type="error",
            workflow_used=False,
            decision_info={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    
    try:
        if orchestrator and orchestrator.is_initialized:
            health_data = await orchestrator.get_system_status()
            
            return HealthResponse(
                status="healthy",
                nava_initialized=True,
                available_models=health_data.get("available_models", ["gpt", "claude", "gemini"]),
                uptime_seconds=float(health_data.get("uptime_seconds", 0)),
                timestamp=datetime.now().isoformat()
            )
        else:
            return HealthResponse(
                status="initializing",
                nava_initialized=False,
                available_models=[],
                uptime_seconds=0.0,
                timestamp=datetime.now().isoformat()
            )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="error",
            nava_initialized=False,
            available_models=[],
            uptime_seconds=0.0,
            timestamp=datetime.now().isoformat()
        )

# Available models endpoint
@app.get("/models")
async def get_available_models():
    """Get available models"""
    
    try:
        if orchestrator:
            return {
                "available_models": ["gpt", "claude", "gemini"],
                "status": "mock",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "available_models": [],
                "status": "not_initialized",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return {
            "available_models": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "service": "NAVA Logic Controller",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health", 
            "models": "/models",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENVIRONMENT") == "development" else False,
        log_level="info"
    )