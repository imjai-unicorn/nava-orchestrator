# backend/shared/utils/validation.py
"""
Validation Utilities for NAVA
Input validation, data sanitization, and security checks
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from email.utils import parseaddr

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputValidator:
    """Comprehensive input validation utilities"""
    
    # Common patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?[\d\s\-\(\)]{10,15}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    SLUG_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    
    # Security patterns
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE)
    ]
    
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)", re.IGNORECASE),
        re.compile(r"['\"];?\s*(DROP|DELETE|INSERT|UPDATE)", re.IGNORECASE),
        re.compile(r"1\s*=\s*1", re.IGNORECASE),
        re.compile(r"'\s*OR\s*'", re.IGNORECASE)
    ]
    
    @staticmethod
    def validate_string(value: Any, min_length: int = 0, max_length: int = 1000, 
                       required: bool = True, pattern: Optional[str] = None) -> str:
        """Validate string input"""
        if value is None:
            if required:
                raise ValidationError("Value is required")
            return ""
        
        if not isinstance(value, str):
            value = str(value)
        
        # Length validation
        if len(value) < min_length:
            raise ValidationError(f"Value must be at least {min_length} characters")
        
        if len(value) > max_length:
            raise ValidationError(f"Value must not exceed {max_length} characters")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            raise ValidationError("Value does not match required pattern")
        
        return value.strip()
    
    @staticmethod
    def validate_email(email: Any) -> str:
        """Validate email address"""
        if not email:
            raise ValidationError("Email is required")
        
        email = str(email).strip().lower()
        
        if not InputValidator.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")
        
        # Additional validation using email.utils
        name, addr = parseaddr(email)
        if not addr:
            raise ValidationError("Invalid email address")
        
        return email
    
    @staticmethod
    def validate_phone(phone: Any) -> str:
        """Validate phone number"""
        if not phone:
            raise ValidationError("Phone number is required")
        
        phone = str(phone).strip()
        
        if not InputValidator.PHONE_PATTERN.match(phone):
            raise ValidationError("Invalid phone number format")
        
        return phone
    
    @staticmethod
    def validate_uuid(uuid_str: Any) -> str:
        """Validate UUID format"""
        if not uuid_str:
            raise ValidationError("UUID is required")
        
        uuid_str = str(uuid_str).strip()
        
        if not InputValidator.UUID_PATTERN.match(uuid_str):
            raise ValidationError("Invalid UUID format")
        
        return uuid_str
    
    @staticmethod
    def validate_integer(value: Any, min_value: Optional[int] = None, 
                        max_value: Optional[int] = None) -> int:
        """Validate integer input"""
        try:
            if isinstance(value, str):
                value = int(value.strip())
            elif not isinstance(value, int):
                value = int(value)
        except (ValueError, TypeError):
            raise ValidationError("Value must be a valid integer")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"Value must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"Value must not exceed {max_value}")
        
        return value
    
    @staticmethod
    def validate_float(value: Any, min_value: Optional[float] = None, 
                      max_value: Optional[float] = None) -> float:
        """Validate float input"""
        try:
            if isinstance(value, str):
                value = float(value.strip())
            elif not isinstance(value, (int, float)):
                value = float(value)
        except (ValueError, TypeError):
            raise ValidationError("Value must be a valid number")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"Value must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"Value must not exceed {max_value}")
        
        return float(value)
    
    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """Validate boolean input"""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ('true', '1', 'yes', 'on'):
                return True
            elif value in ('false', '0', 'no', 'off'):
                return False
        
        if isinstance(value, int):
            return bool(value)
        
        raise ValidationError("Value must be a valid boolean")
    
    @staticmethod
    def validate_datetime(value: Any, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Validate datetime input"""
        if isinstance(value, datetime):
            return value
        
        if not isinstance(value, str):
            raise ValidationError("Datetime must be a string or datetime object")
        
        try:
            return datetime.strptime(value.strip(), format_str)
        except ValueError:
            raise ValidationError(f"Invalid datetime format. Expected: {format_str}")
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any]) -> Any:
        """Validate value against allowed choices"""
        if value not in choices:
            raise ValidationError(f"Value must be one of: {', '.join(map(str, choices))}")
        
        return value
    
    @staticmethod
    def validate_json(value: Any) -> Dict[str, Any]:
        """Validate JSON input"""
        if isinstance(value, dict):
            return value
        
        if not isinstance(value, str):
            raise ValidationError("JSON must be a string or dictionary")
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")

class SecurityValidator:
    """Security-focused validation"""
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Basic HTML sanitization"""
        if not text:
            return ""
        
        # Remove potential XSS patterns
        for pattern in InputValidator.XSS_PATTERNS:
            text = pattern.sub('', text)
        
        # Escape remaining HTML characters
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('"', '&quot;').replace("'", '&#x27;')
        
        return text
    
    @staticmethod
    def check_sql_injection(text: str) -> bool:
        """Check for potential SQL injection patterns"""
        if not text:
            return False
        
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        
        return False
    
    @staticmethod
    def validate_file_upload(filename: str, allowed_extensions: List[str],
                           max_size_mb: int = 10) -> bool:
        """Validate file upload"""
        if not filename:
            raise ValidationError("Filename is required")
        
        # Check extension
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if file_ext not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(f"File type not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            re.compile(r'\.php$', re.IGNORECASE),
            re.compile(r'\.exe$', re.IGNORECASE),
            re.compile(r'\.bat$', re.IGNORECASE),
            re.compile(r'\.sh$', re.IGNORECASE),
            re.compile(r'\.js$', re.IGNORECASE),
        ]
        
        for pattern in suspicious_patterns:
            if pattern.search(filename):
                raise ValidationError("Potentially dangerous file type")
        
        return True
    
    @staticmethod
    def validate_password_strength(password: str, min_length: int = 8) -> Dict[str, bool]:
        """Validate password strength"""
        if not password:
            raise ValidationError("Password is required")
        
        checks = {
            'length': len(password) >= min_length,
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'lowercase': bool(re.search(r'[a-z]', password)),
            'digit': bool(re.search(r'\d', password)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        
        score = sum(checks.values())
        checks['strong'] = score >= 4
        
        return checks

class DataValidator:
    """Data structure validation"""
    
    @staticmethod
    def validate_dict(data: Any, required_keys: List[str] = None,
                     optional_keys: List[str] = None) -> Dict[str, Any]:
        """Validate dictionary structure"""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary")
        
        if required_keys:
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValidationError(f"Missing required keys: {', '.join(missing_keys)}")
        
        if optional_keys is not None:
            all_allowed_keys = set(required_keys or []) | set(optional_keys)
            extra_keys = set(data.keys()) - all_allowed_keys
            if extra_keys:
                raise ValidationError(f"Unexpected keys: {', '.join(extra_keys)}")
        
        return data
    
    @staticmethod
    def validate_list(data: Any, min_length: int = 0, max_length: int = 1000,
                     item_validator: Optional[Callable] = None) -> List[Any]:
        """Validate list structure"""
        if not isinstance(data, list):
            raise ValidationError("Data must be a list")
        
        if len(data) < min_length:
            raise ValidationError(f"List must have at least {min_length} items")
        
        if len(data) > max_length:
            raise ValidationError(f"List must not exceed {max_length} items")
        
        if item_validator:
            validated_items = []
            for i, item in enumerate(data):
                try:
                    validated_items.append(item_validator(item))
                except ValidationError as e:
                    raise ValidationError(f"Item {i}: {e}")
            return validated_items
        
        return data

class NAVAValidator:
    """NAVA-specific validation"""
    
    @staticmethod
    def validate_ai_model_name(model: str) -> str:
        """Validate AI model name"""
        allowed_models = ['gpt', 'claude', 'gemini', 'local', 'phi3', 'deepseek']
        
        model = InputValidator.validate_string(model, min_length=2, max_length=20)
        
        if model not in allowed_models:
            raise ValidationError(f"Invalid AI model. Allowed: {', '.join(allowed_models)}")
        
        return model
    
    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """Validate session ID format"""
        return InputValidator.validate_uuid(session_id)
    
    @staticmethod
    def validate_user_message(message: str) -> str:
        """Validate user message content"""
        message = InputValidator.validate_string(
            message, min_length=1, max_length=10000, required=True
        )
        
        # Security checks
        if SecurityValidator.check_sql_injection(message):
            raise ValidationError("Message contains suspicious content")
        
        # Sanitize HTML
        message = SecurityValidator.sanitize_html(message)
        
        return message
    
    @staticmethod
    def validate_workflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow configuration"""
        required_keys = ['name', 'steps']
        optional_keys = ['description', 'timeout', 'priority', 'retry_count']
        
        config = DataValidator.validate_dict(config, required_keys, optional_keys)
        
        # Validate specific fields
        config['name'] = InputValidator.validate_string(config['name'], min_length=1, max_length=100)
        
        if 'timeout' in config:
            config['timeout'] = InputValidator.validate_integer(config['timeout'], min_value=1, max_value=3600)
        
        if 'priority' in config:
            config['priority'] = InputValidator.validate_choice(config['priority'], [1, 2, 3, 4, 5])
        
        return config
    
    @staticmethod
    def validate_api_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API request data"""
        # Common API validation
        if 'user_id' in data and data['user_id']:
            data['user_id'] = InputValidator.validate_string(data['user_id'], max_length=50)
        
        if 'session_id' in data and data['session_id']:
            data['session_id'] = NAVAValidator.validate_session_id(data['session_id'])
        
        if 'message' in data:
            data['message'] = NAVAValidator.validate_user_message(data['message'])
        
        if 'ai_model' in data and data['ai_model']:
            data['ai_model'] = NAVAValidator.validate_ai_model_name(data['ai_model'])
        
        return data

# Convenience functions
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Quick validation of required fields"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

def sanitize_input(text: str) -> str:
    """Quick input sanitization"""
    if not text:
        return ""
    
    # Basic sanitization
    text = SecurityValidator.sanitize_html(text)
    text = text.strip()
    
    return text

def validate_and_sanitize_chat_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize chat input"""
    validated_data = {}
    
    # Required fields
    validate_required_fields(data, ['message'])
    
    # Validate and sanitize
    validated_data['message'] = NAVAValidator.validate_user_message(data['message'])
    
    if 'session_id' in data and data['session_id']:
        validated_data['session_id'] = NAVAValidator.validate_session_id(data['session_id'])
    
    if 'user_id' in data and data['user_id']:
        validated_data['user_id'] = InputValidator.validate_string(data['user_id'], max_length=50)
    
    if 'ai_model' in data and data['ai_model']:
        validated_data['ai_model'] = NAVAValidator.validate_ai_model_name(data['ai_model'])
    
    if 'preferences' in data and data['preferences']:
        validated_data['preferences'] = InputValidator.validate_json(data['preferences'])
    
    return validated_data

# Export main classes and functions
__all__ = [
    'ValidationError',
    'InputValidator', 
    'SecurityValidator',
    'DataValidator',
    'NAVAValidator',
    'validate_required_fields',
    'sanitize_input',
    'validate_and_sanitize_chat_input'
]
