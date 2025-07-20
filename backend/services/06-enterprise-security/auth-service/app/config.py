# backend/services/06-enterprise-security/auth-service/app/config.py
"""
Authentication Service Configuration
Centralized configuration management
"""

import os
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import validator
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class AuthConfig(BaseSettings):
    """Authentication service configuration"""
    
    # Service Configuration
    port: int = 8007
    service_name: str = "auth-service"
    environment: str = "production"
    debug_mode: bool = False
    reload_on_change: bool = False
    
    # JWT Configuration
    jwt_secret_key: str = "nava-enterprise-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters long')
        return v
    
    # Password Security
    bcrypt_rounds: int = 12
    password_min_length: int = 8
    password_require_special: bool = True
    
    # MFA Configuration
    mfa_issuer: str = "NAVA Enterprise"
    mfa_backup_codes_count: int = 8
    mfa_totp_validity_window: int = 1
    
    # Session Management
    max_sessions_per_user: int = 5
    session_timeout_hours: int = 8
    security_timeout_minutes: int = 30
    
    # Account Security
    max_login_attempts: int = 5
    account_lockout_minutes: int = 30
    password_reset_token_expire_minutes: int = 15
    
    # Database Configuration (Optional)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Email Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    
    # Enterprise SSO Configuration (Optional)
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_x509_cert: Optional[str] = None
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Security Configuration
    security_headers_enabled: bool = True
    cors_origins: str = "*"
    cors_credentials: bool = True
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # Audit Configuration
    audit_log_enabled: bool = True
    audit_log_level: str = "INFO"
    security_event_retention_days: int = 365
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefix
        env_prefix = ""
        
        # Allow extra fields
        extra = "ignore"

class SecurityConfig:
    """Security-specific configuration"""
    
    # Password validation patterns
    PASSWORD_PATTERNS = {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digit": True,
        "require_special": True,
        "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?"
    }
    
    # Session security levels
    SESSION_SECURITY_LEVELS = {
        "standard": {
            "timeout_hours": 8,
            "require_mfa": False,
            "ip_binding": False
        },
        "elevated": {
            "timeout_hours": 4,
            "require_mfa": True,
            "ip_binding": True
        },
        "critical": {
            "timeout_hours": 1,
            "require_mfa": True,
            "ip_binding": True
        }
    }
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

class DatabaseConfig:
    """Database configuration"""
    
    # Connection settings
    CONNECTION_TIMEOUT = 30
    QUERY_TIMEOUT = 30
    MAX_CONNECTIONS = 20
    MIN_CONNECTIONS = 5
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    BACKOFF_FACTOR = 2.0

# Initialize configuration
def get_config() -> AuthConfig:
    """Get application configuration"""
    config = AuthConfig()
    
    # Log configuration (but mask sensitive values)
    config_dict = config.dict()
    sensitive_keys = ['jwt_secret_key', 'smtp_password', 'database_url', 'redis_url']
    
    for key in sensitive_keys:
        if config_dict.get(key):
            config_dict[key] = "***MASKED***"
    
    logger.info(f"🔧 Authentication service configuration loaded: {config_dict}")
    
    return config

# Global configuration instance
config = get_config()

# Configuration validation
def validate_config(config: AuthConfig) -> bool:
    """Validate configuration"""
    errors = []
    
    # Check required fields for production
    if config.environment == "production":
        if config.jwt_secret_key == "nava-enterprise-secret-key-change-in-production":
            errors.append("JWT secret key must be changed in production")
        
        if not config.smtp_username and config.audit_log_enabled:
            errors.append("SMTP configuration required for audit notifications")
    
    # Check password policy
    if config.password_min_length < 8:
        errors.append("Password minimum length should be at least 8 characters")
    
    # Check session timeout
    if config.session_timeout_hours > 24:
        errors.append("Session timeout should not exceed 24 hours")
    
    if errors:
        logger.error(f"❌ Configuration validation failed: {errors}")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

# Export commonly used configurations
JWT_SECRET_KEY = config.jwt_secret_key
JWT_ALGORITHM = config.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = config.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = config.refresh_token_expire_days

# Session configuration
MAX_SESSIONS_PER_USER = config.max_sessions_per_user
SESSION_TIMEOUT_HOURS = config.session_timeout_hours

# Security configuration
MAX_LOGIN_ATTEMPTS = config.max_login_attempts
ACCOUNT_LOCKOUT_MINUTES = config.account_lockout_minutes

# MFA configuration
MFA_ISSUER = config.mfa_issuer
MFA_BACKUP_CODES_COUNT = config.mfa_backup_codes_count