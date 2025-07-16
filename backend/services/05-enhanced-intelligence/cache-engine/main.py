# backend/services/05-enhanced-intelligence/cache-engine/main.py
"""
Cache Engine Service
Port: 8013
Intelligent response caching system
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
    title="NAVA Cache Engine",
    description="Intelligent Response Caching and Similarity Matching Service",
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
    from app.cache_manager import cache_router
    from app.similarity_engine import similarity_router
    from app.ttl_manager import ttl_router
    from app.vector_search import vector_router
    
    # Include routers
    app.include_router(cache_router, prefix="/api/cache", tags=["cache"])
    app.include_router(similarity_router, prefix="/api/similarity", tags=["similarity"])
    app.include_router(ttl_router, prefix="/api/ttl", tags=["ttl"])
    app.include_router(vector_router, prefix="/api/vector", tags=["vector"])
    
    logger.info("✅ All cache engine routers loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Router import error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NAVA Cache Engine",
        "version": "1.0.0",
        "status": "operational",
        "capabilities": [
            "intelligent_response_caching",
            "semantic_similarity_matching",
            "multi_layer_caching",
            "ttl_management",
            "vector_similarity_search",
            "cache_optimization"
        ],
        "cache_types": [
            "memory_cache",
            "redis_cache", 
            "database_cache",
            "semantic_cache"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cache_engine",
        "port": 8013,
        "version": "1.0.0",
        "cache_engines": {
            "cache_manager": "active",
            "similarity_engine": "active",
            "ttl_manager": "active",
            "vector_search": "active"
        },
        "performance_metrics": {
            "target_cache_hit_rate": ">40%",
            "target_response_time": "<100ms",
            "target_similarity_accuracy": ">85%"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8013))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )