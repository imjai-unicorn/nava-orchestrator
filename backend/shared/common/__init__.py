# backend/services/shared/common/__init__.py
"""NAVA Common Utilities Module"""

# Import Week 1.5 critical components
try:
    from .circuit_breaker import (
        EnhancedCircuitBreaker, 
        circuit_breaker, 
        enhanced_circuit_breaker,
        call_ai_with_protection,
        get_service_health,
        reset_service_circuit
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Circuit breaker not available: {e}")
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from .cache_manager import (
        IntelligentCacheManager,
        global_cache,
        cache_manager,
        get_cached_response,
        cache_response,
        get_cache_stats
    )
    CACHE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Cache manager not available: {e}")
    CACHE_MANAGER_AVAILABLE = False

# Make available at package level
__all__ = [
    # Circuit Breaker
    'EnhancedCircuitBreaker',
    'circuit_breaker', 
    'enhanced_circuit_breaker',
    'call_ai_with_protection',
    'get_service_health',
    'reset_service_circuit',
    'CIRCUIT_BREAKER_AVAILABLE',
    
    # Cache Manager
    'IntelligentCacheManager',
    'global_cache',
    'cache_manager',
    'get_cached_response', 
    'cache_response',
    'get_cache_stats',
    'CACHE_MANAGER_AVAILABLE'
]