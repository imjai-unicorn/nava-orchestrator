# backend/services/05-enhanced-intelligence/decision-engine/main.py
"""
Enhanced Decision Engine Service
Port: 8008
Pure Microservice for AI Decision Making
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NAVA Decision Engine",
    description="Enhanced AI Decision Making Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import routers
try:
    from app.enhanced_decision_engine import enhanced_decision_router
    from app.criteria_analyzer import criteria_router
    from app.outcome_predictor import outcome_router
    from app.risk_assessor import risk_router
    
    # Include routers
    app.include_router(enhanced_decision_router, prefix="/api/decision", tags=["decision"])
    app.include_router(criteria_router, prefix="/api/criteria", tags=["criteria"])
    app.include_router(outcome_router, prefix="/api/outcome", tags=["outcome"])
    app.include_router(risk_router, prefix="/api/risk", tags=["risk"])
    
    logger.info("✅ Enhanced decision engine router loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Router import error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NAVA Decision Engine",
        "version": "1.0.0",
        "status": "operational",
        "capabilities": [
            "enhanced_ai_selection",
            "criteria_analysis", 
            "outcome_prediction",
            "risk_assessment",
            "decision_transparency"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "decision_engine",
        "port": 8008,
        "version": "1.0.0",
        "uptime": "operational",
        "capabilities_status": {
            "decision_analysis": "active",
            "criteria_evaluation": "active",
            "outcome_prediction": "active",
            "risk_assessment": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

# ใน main.py - line สุดท้าย
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8008))  # ✅ ถูกแล้ว
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )