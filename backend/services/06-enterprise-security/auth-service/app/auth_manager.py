# backend/services/06-enterprise-security/auth-service/app/auth_manager.py
"""
Authentication Manager
Core authentication logic and user management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import bcrypt
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# User roles enum
class UserRole(str, Enum):
    ADMIN = "admin"
    ENTERPRISE_USER = "enterprise_user"
    DEVELOPER = "developer"
    AUDITOR = "auditor"
    READONLY = "readonly"

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.ENTERPRISE_USER
    department: Optional[str] = None
    requires_mfa: bool = True

class UserLogin(BaseModel):
    username: str
    password: str
    mfa_code: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: UserRole
    department: Optional[str]
    requires_mfa: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        "user:create", "user:read", "user:update", "user:delete",
        "system:admin", "audit:read", "config:update"
    ],
    UserRole.ENTERPRISE_USER: [
        "nava:chat", "nava:models", "profile:read", "profile:update"
    ],
    UserRole.DEVELOPER: [
        "nava:chat", "nava:models", "nava:debug", "api:test", 
        "logs:read", "metrics:read"
    ],
    UserRole.AUDITOR: [
        "audit:read", "logs:read", "metrics:read", "compliance:read"
    ],
    UserRole.READONLY: [
        "nava:read", "profile:read"
    ]
}

class AuthManager:
    def __init__(self):
        self.users_db = {}  # In production, use proper database
        self.sessions = {}
        logger.info("🔐 Authentication Manager initialized")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user"""
        # Check if user exists
        if user_data.username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Hash password
        hashed_password = self.hash_password(user_data.password)
        
        # Create user record
        user_id = f"user_{len(self.users_db) + 1}"
        user_record = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hashed_password,
            "role": user_data.role,
            "department": user_data.department,
            "requires_mfa": user_data.requires_mfa,
            "is_active": True,
            "created_at": datetime.now(),
            "last_login": None,
            "failed_login_attempts": 0,
            "locked_until": None
        }
        
        self.users_db[user_data.username] = user_record
        
        logger.info(f"✅ User created: {user_data.username} with role {user_data.role}")
        
        return UserResponse(
            user_id=user_id,
            username=user_data.username,
            email=user_data.email,
            role=user_data.role,
            department=user_data.department,
            requires_mfa=user_data.requires_mfa,
            is_active=True,
            created_at=user_record["created_at"],
            last_login=None
        )

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password"""
        user = self.users_db.get(username)
        if not user:
            return None

        # Check if account is locked
        if user.get("locked_until") and datetime.now() < user["locked_until"]:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to failed login attempts"
            )

        # Verify password
        if not self.verify_password(password, user["password_hash"]):
            # Increment failed attempts
            user["failed_login_attempts"] = user.get("failed_login_attempts", 0) + 1
            
            # Lock account after 5 failed attempts
            if user["failed_login_attempts"] >= 5:
                user["locked_until"] = datetime.now() + timedelta(minutes=30)
                logger.warning(f"🔒 Account locked: {username}")
            
            return None

        # Reset failed attempts on successful authentication
        user["failed_login_attempts"] = 0
        user["locked_until"] = None
        user["last_login"] = datetime.now()
        
        return user

    def get_user_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for user role"""
        return ROLE_PERMISSIONS.get(role, [])

# Initialize auth manager
auth_manager = AuthManager()

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register new user"""
    return auth_manager.create_user(user_data)

@router.post("/login", response_model=AuthResponse)
async def login_user(login_data: UserLogin):
    """User login"""
    # Authenticate user
    user = auth_manager.authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Check MFA if required
    if user["requires_mfa"] and not login_data.mfa_code:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MFA code required"
        )

    # TODO: Verify MFA code if provided
    if user["requires_mfa"] and login_data.mfa_code:
        # MFA verification will be implemented in mfa_handler
        pass

    # Generate JWT tokens (will be implemented in jwt_handler)
    from .jwt_handler import create_access_token, create_refresh_token
    
    # Get user permissions
    permissions = auth_manager.get_user_permissions(user["role"])
    
    # Create tokens
    access_token = create_access_token(
        data={
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "permissions": permissions
        }
    )
    
    refresh_token = create_refresh_token(data={"user_id": user["user_id"]})
    
    logger.info(f"✅ User logged in: {user['username']}")
    
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=3600,  # 1 hour
        user=UserResponse(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            role=user["role"],
            department=user["department"],
            requires_mfa=user["requires_mfa"],
            is_active=user["is_active"],
            created_at=user["created_at"],
            last_login=user["last_login"]
        )
    )

@router.get("/users", response_model=List[UserResponse])
async def list_users(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """List all users (admin only)"""
    # TODO: Verify admin permissions
    users = []
    for user_data in auth_manager.users_db.values():
        users.append(UserResponse(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            department=user_data["department"],
            requires_mfa=user_data["requires_mfa"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"]
        ))
    
    return users

@router.get("/permissions/{role}")
async def get_role_permissions(role: UserRole):
    """Get permissions for a role"""
    return {
        "role": role,
        "permissions": auth_manager.get_user_permissions(role)
    }

# Export router
auth_router = router