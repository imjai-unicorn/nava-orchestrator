import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="NAVA Authentication Service",
    description="Enterprise Authentication and Authorization Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "auth-service",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "NAVA Authentication Service",
        "status": "running",
        "docs": "/docs"
    }

# Auth status endpoint
@app.get("/auth/status")
async def auth_status():
    return {
        "auth_service": "operational",
        "features": [
            "JWT Authentication",
            "Multi-factor Authentication", 
            "Session Management",
            "Role-based Access Control"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8007))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )