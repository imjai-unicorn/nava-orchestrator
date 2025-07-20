# backend/services/06-enterprise-security/auth-service/app/jwt_handler.py
"""
JWT Token Handler
JWT token creation, validation, and management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "nava-enterprise-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30    # 30 days

# Pydantic models
class TokenData(BaseModel):
    user_id: str
    username: str
    role: str
    permissions: list[str]

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenValidation(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[list[str]] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

class JWTHandler:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.revoked_tokens = set()  # In production, use Redis or database
        logger.info("🔑 JWT Handler initialized")

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"✅ Access token created for user: {data.get('username')}")
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"✅ Refresh token created for user: {data.get('user_id')}")
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            token_type = payload.get("type")
            if not token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Create new access token from refresh token"""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )

            user_id = payload.get("user_id")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token payload"
                )

            # Get user data (in production, fetch from database)
            # For now, create minimal token data
            token_data = {
                "user_id": user_id,
                "username": f"user_{user_id}",  # Would fetch from DB
                "role": "enterprise_user",       # Would fetch from DB
                "permissions": ["nava:chat", "nava:models"]  # Would fetch from DB
            }

            # Create new access token
            new_access_token = self.create_access_token(token_data)
            new_refresh_token = self.create_refresh_token({"user_id": user_id})

            logger.info(f"✅ Token refreshed for user: {user_id}")

            return TokenResponse(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )

        except Exception as e:
            logger.error(f"❌ Token refresh failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token refresh failed"
            )

    def revoke_token(self, token: str) -> bool:
        """Revoke token (logout)"""
        try:
            # Verify token first
            payload = self.verify_token(token)
            
            # Add to revoked tokens
            self.revoked_tokens.add(token)
            
            logger.info(f"✅ Token revoked for user: {payload.get('username')}")
            return True

        except Exception as e:
            logger.error(f"❌ Token revocation failed: {e}")
            return False

# Initialize JWT handler
jwt_handler = JWTHandler()

# Export functions for use in other modules
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    return jwt_handler.create_access_token(data, expires_delta)

def create_refresh_token(data: Dict[str, Any]) -> str:
    return jwt_handler.create_refresh_token(data)

def verify_jwt_token(token: str) -> Dict[str, Any]:
    return jwt_handler.verify_token(token)

# API Routes
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token"""
    return jwt_handler.refresh_access_token(request.refresh_token)

@router.post("/validate", response_model=TokenValidation)
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token"""
    try:
        token = credentials.credentials
        payload = jwt_handler.verify_token(token)
        
        return TokenValidation(
            valid=True,
            user_id=payload.get("user_id"),
            username=payload.get("username"),
            role=payload.get("role"),
            permissions=payload.get("permissions", []),
            expires_at=datetime.fromtimestamp(payload.get("exp", 0))
        )
    
    except HTTPException as e:
        return TokenValidation(
            valid=False,
            error=e.detail
        )
    except Exception as e:
        return TokenValidation(
            valid=False,
            error="Token validation failed"
        )

@router.post("/revoke")
async def revoke_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Revoke token (logout)"""
    token = credentials.credentials
    success = jwt_handler.revoke_token(token)
    
    if success:
        return {"message": "Token revoked successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token revocation failed"
        )

@router.get("/info")
async def token_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get token information"""
    token = credentials.credentials
    payload = jwt_handler.verify_token(token)
    
    return {
        "user_id": payload.get("user_id"),
        "username": payload.get("username"),
        "role": payload.get("role"),
        "permissions": payload.get("permissions", []),
        "token_type": payload.get("type"),
        "issued_at": datetime.fromtimestamp(payload.get("iat", 0)),
        "expires_at": datetime.fromtimestamp(payload.get("exp", 0))
    }

# Export router
jwt_router = router