"""NAVA Shared Components Package

Shared utilities and common services for NAVA microservices.
Provides cross-service functionality including caching, circuit breaking,
and common utilities.

Version: 1.0.0
Status: Core Infrastructure
"""

__version__ = "1.0.0"
__author__ = "NAVA Development Team"

from .common.circuit_breaker import EnhancedCircuitBreaker
from .common.cache_manager import ResponseCache
# Import core shared components
# try:
#    from .common.circuit_breaker import EnhancedCircuitBreaker
#    from .common.cache_manager import ResponseCache

#    SHARED_COMPONENTS_AVAILABLE = True
#    print("✅ Shared components loaded successfully")
#except ImportError as e:
#    print(f"⚠️ Shared components not available: {e}")
#    EnhancedCircuitBreaker = None
#    ResponseCache = None
#    SHARED_COMPONENTS_AVAILABLE = False

__all__ = [
    "EnhancedCircuitBreaker",
    "ResponseCache",
    "SHARED_COMPONENTS_AVAILABLE"
]
