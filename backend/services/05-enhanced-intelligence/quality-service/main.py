# backend/services/05-enhanced-intelligence/quality-service/main.py
"""
Quality Service
Port: 8009
Pure Microservice for Response Quality Validation
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
    title="NAVA Quality Service",
    description="Advanced Response Quality Validation Service",
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
    from app.quality_validator import quality_router
    from app.response_scorer import scorer_router
    from app.improvement_suggester import improvement_router
    
    # Include routers
    app.include_router(quality_router, prefix="/api/quality", tags=["quality"])
    app.include_router(scorer_router, prefix="/api/scorer", tags=["scorer"])
    app.include_router(improvement_router, prefix="/api/improvement", tags=["improvement"])
    
    logger.info("✅ All quality service routers loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Router import error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NAVA Quality Service",
        "version": "1.0.0",
        "status": "operational",
        "capabilities": [
            "response_quality_validation",
            "multi_dimensional_scoring",
            "improvement_suggestions",
            "enterprise_compliance",
            "quality_monitoring"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "quality_service",
        "port": 8009,
        "version": "1.0.0",
        "quality_engines": {
            "validator": "active",
            "scorer": "active",
            "improvement_suggester": "active"
        },
        "thresholds": {
            "minimum_quality": 0.75,
            "safety_threshold": 0.95,
            "compliance_threshold": 0.90
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8009))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )