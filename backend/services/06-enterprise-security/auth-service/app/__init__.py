# backend/services/06-enterprise-security/auth-service/__init__.py
"""
NAVA Enterprise Authentication Service
Multi-factor authentication, RBAC, and enterprise security
"""

__version__ = "1.0.0"
__author__ = "NAVA Enterprise Team"
__description__ = "Enterprise Authentication and Authorization Service"

# backend/services/06-enterprise-security/auth-service/app/__init__.py
"""
Authentication Service Application Module
"""

from .config import config, get_config, validate_config
from .auth_manager import auth_manager, auth_router
from .jwt_handler import jwt_handler, jwt_router
from .mfa_handler import mfa_handler, mfa_router
from .session_manager import session_manager, session_router

__all__ = [
    "config",
    "get_config", 
    "validate_config",
    "auth_manager",
    "auth_router",
    "jwt_handler", 
    "jwt_router",
    "mfa_handler",
    "mfa_router",
    "session_manager",
    "session_router"
]