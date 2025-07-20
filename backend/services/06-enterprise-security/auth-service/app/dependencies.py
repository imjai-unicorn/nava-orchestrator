# backend/services/06-enterprise-security/auth-service/app/dependencies.py
"""
Authentication Service Dependencies
Dependency injection for FastAPI
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from .config import config
from .jwt_handler import jwt_handler
from .auth_manager import auth_manager
from .session_manager import session_manager

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

class AuthenticationDependency:
    """Authentication dependency for protecting routes"""
    
    def __init__(self, required_permissions: Optional[List[str]] = None,
                 required_role: Optional[str] = None,
                 require_mfa: bool = False):
        self.required_permissions = required_permissions or []
        self.required_role = required_role
        self.require_mfa = require_mfa

    async def __call__(self, 
                      request: Request,
                      credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
        """Validate authentication and authorization"""
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            # Verify JWT token
            payload = jwt_handler.verify_token(credentials.credentials)
            
            # Extract user information
            user_id = payload.get("user_id")
            username = payload.get("username")
            role = payload.get("role")
            permissions = payload.get("permissions", [])
            
            if not user_id or not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )

            # Check role requirement
            if self.required_role and role != self.required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{self.required_role}' required"
                )

            # Check permission requirements
            if self.required_permissions:
                missing_permissions = set(self.required_permissions) - set(permissions)
                if missing_permissions:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing permissions: {list(missing_permissions)}"
                    )

            # Check MFA requirement (would integrate with session management)
            if self.require_mfa:
                # In production, check if user completed MFA for this session
                pass

            # Log access attempt
            logger.info(f"✅ Authenticated access: {username} ({role}) from {request.client.host}")

            return {
                "user_id": user_id,
                "username": username,
                "role": role,
                "permissions": permissions,
                "token_payload": payload
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

# Dependency functions for common use cases
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    auth_dep = AuthenticationDependency()
    # We need to create a mock request for this dependency
    from fastapi import Request
    mock_request = type('MockRequest', (), {'client': type('Client', (), {'host': '127.0.0.1'})()})()
    return await auth_dep(mock_request, credentials)

async def require_admin(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Require admin role"""
    auth_dep = AuthenticationDependency(required_role="admin")
    return await auth_dep(request, credentials)

async def require_developer(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Require developer role"""
    auth_dep = AuthenticationDependency(
        required_permissions=["nava:debug", "api:test"],
        required_role="developer"
    )
    return await auth_dep(request, credentials)

async def require_auditor(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Require auditor role"""
    auth_dep = AuthenticationDependency(
        required_permissions=["audit:read", "logs:read"],
        required_role="auditor"
    )
    return await auth_dep(request, credentials)

async def require_enterprise_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Require enterprise user role or higher"""
    auth_dep = AuthenticationDependency(
        required_permissions=["nava:chat"]
    )
    return await auth_dep(request, credentials)

# Rate limiting dependency
class RateLimitDependency:
    """Rate limiting dependency"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history = {}  # In production, use Redis
    
    async def __call__(self, request: Request) -> bool:
        """Check rate limit"""
        if not config.rate_limit_enabled:
            return True
        
        client_ip = request.client.host
        current_time = datetime.now()
        
        # Clean old entries (simple implementation)
        cutoff_time = current_time.timestamp() - 60  # 1 minute ago
        
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        # Remove old requests
        self.request_history[client_ip] = [
            timestamp for timestamp in self.request_history[client_ip]
            if timestamp > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(self.request_history[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.request_history[client_ip].append(current_time.timestamp())
        
        return True

# Initialize rate limiter
rate_limiter = RateLimitDependency(config.rate_limit_requests_per_minute)

# Security headers dependency
async def add_security_headers(request: Request, response):
    """Add security headers to response"""
    if config.security_headers_enabled:
        from .config import SecurityConfig
        
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
    
    return response

# Session validation dependency
async def validate_session(request: Request, 
                          user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Validate active session"""
    user_id = user["user_id"]
    
    # Get active sessions for user
    active_sessions = session_manager.get_user_sessions(user_id, active_only=True)
    
    if not active_sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active session found"
        )
    
    # Update session activity
    # In a real implementation, we'd track session ID in the JWT
    # For now, just update the most recent session
    if active_sessions:
        latest_session = active_sessions[0]
        session_manager.update_session_activity(
            latest_session["session_id"],
            f"API access: {request.url.path}"
        )
    
    return user

# Audit logging dependency
class AuditLogDependency:
    """Audit logging dependency"""
    
    async def __call__(self, 
                      request: Request,
                      user: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        if not config.audit_log_enabled:
            return
        
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "user_id": user.get("user_id") if user else None,
            "username": user.get("username") if user else None,
            "role": user.get("role") if user else None
        }
        
        # In production, send to audit logging service
        logger.info(f"🔍 AUDIT: {audit_event}")

audit_logger = AuditLogDependency()

# Optional authentication (for public endpoints that benefit from user context)
async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        payload = jwt_handler.verify_token(credentials.credentials)
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "role": payload.get("role"),
            "permissions": payload.get("permissions", [])
        }
    except Exception:
        return None

# Database dependency (for future use)
async def get_database():
    """Get database connection"""
    # Placeholder for database connection
    # In production, this would return actual database session
    return None

# Cache dependency (for future use)
async def get_cache():
    """Get cache connection"""
    # Placeholder for Redis/cache connection
    # In production, this would return actual cache client
    return None

# Export commonly used dependencies
RequireAuthentication = Depends(get_current_user)
RequireAdmin = Depends(require_admin)
RequireDeveloper = Depends(require_developer)
RequireAuditor = Depends(require_auditor)
RequireEnterpriseUser = Depends(require_enterprise_user)
RequireValidSession = Depends(validate_session)
RateLimit = Depends(rate_limiter)
AuditLog = Depends(audit_logger)
OptionalAuthentication = Depends(get_optional_user)