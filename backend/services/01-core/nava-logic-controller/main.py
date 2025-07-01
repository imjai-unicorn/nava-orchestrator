 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import from our app
from app.models.chat import ChatRequest, ChatResponse
from app.core.decision_engine import DecisionEngine
from app.services.ai_client import MockAIClient

# Import from shared components
from backend.shared.supabase_client.client import SupabaseManager

from backend.shared.common.config import settings

app = FastAPI(title="NAVA Logic Controller", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
decision_engine = DecisionEngine()
ai_client = MockAIClient()
db = SupabaseManager()

@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Main chat processing endpoint"""
    
    try:
        # 1. Get or create conversation
        if request.conversation_id:
            conversation_id = request.conversation_id
        else:
            conversation_id = await db.save_conversation("New Conversation")
            if not conversation_id:
                raise HTTPException(status_code=500, detail="Failed to create conversation")
        
        # 2. Decide which model to use
        decision = await decision_engine.decide_model(request.message)
        
        # 3. Generate response using selected model
        response = await ai_client.generate_response(
            decision['model'],
            request.message
        )
        
        # 4. Return response
        return ChatResponse(
            response=response,
            model_used=decision['model'],
            confidence=decision['confidence'],
            reasoning=decision['reasoning'],
            conversation_id=conversation_id,
            message_id="mock-message-id"  # For Phase 1
        )
        
    except Exception as e:
        print(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_healthy = db.test_connection()
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "service": "nava-logic-controller",
        "database": "connected" if db_healthy else "disconnected"
    }

@app.get("/")
async def root():
    return {"message": "NAVA Logic Controller - Phase 1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)