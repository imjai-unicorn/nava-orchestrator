# backend/services/06-enterprise-security/auth-service/main.py
"""
Enhanced Authentication Service
Port: 8007
Multi-factor authentication, RBAC, and enterprise SSO
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
import os
from datetime import datetime
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NAVA Enterprise Authentication Service",
    description="Multi-factor Authentication, RBAC, and Enterprise SSO",
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

# Security scheme
security = HTTPBearer()

# Import routers
try:
    from app.auth_manager import auth_router
    from app.jwt_handler import jwt_router
    from app.mfa_handler import mfa_router
    from app.session_manager import session_router
    
    # Include routers
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    app.include_router(jwt_router, prefix="/api/jwt", tags=["jwt"])
    app.include_router(mfa_router, prefix="/api/mfa", tags=["mfa"])
    app.include_router(session_router, prefix="/api/session", tags=["session"])
    
    logger.info("✅ All authentication routers loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Router import error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NAVA Enterprise Authentication Service",
        "version": "1.0.0",
        "status": "operational",
        "capabilities": [
            "multi_factor_authentication",
            "role_based_access_control",
            "enterprise_sso_integration",
            "jwt_token_management",
            "session_management",
            "security_audit_logging"
        ],
        "auth_methods": [
            "username_password",
            "mfa_totp",
            "mfa_sms", 
            "enterprise_sso",
            "api_key"
        ],
        "supported_roles": [
            "admin",
            "enterprise_user",
            "developer",
            "auditor",
            "readonly"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "auth_service",
        "port": 8007,
        "version": "1.0.0",
        "authentication_systems": {
            "auth_manager": "active",
            "jwt_handler": "active",
            "mfa_handler": "active",
            "session_manager": "active"
        },
        "security_features": {
            "multi_factor_auth": "enabled",
            "role_based_access": "enabled",
            "session_security": "enabled",
            "audit_logging": "enabled"
        },
        "performance_targets": {
            "authentication_time": "<500ms",
            "session_validation": "<100ms",
            "mfa_verification": "<2s"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/verify")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Token verification endpoint"""
    try:
        # Import JWT handler for verification
        from app.jwt_handler import verify_jwt_token
        
        token = credentials.credentials
        payload = verify_jwt_token(token)
        
        return {
            "valid": True,
            "user_id": payload.get("user_id"),
            "role": payload.get("role"),
            "permissions": payload.get("permissions", []),
            "expires_at": payload.get("exp")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8007))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )