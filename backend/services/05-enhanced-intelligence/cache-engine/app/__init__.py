# ===== 3. CACHE ENGINE SERVICE (Port 8013) =====

# backend/services/05-enhanced-intelligence/cache-engine/app/__init__.py
"""
Cache Engine Service Package
Multi-layer intelligent caching and similarity matching system
"""

__version__ = "1.0.0"
__service__ = "cache-engine"

from .cache_manager import cache_router, cache_manager
from .similarity_engine import similarity_router, similarity_engine
from .ttl_manager import ttl_router, ttl_manager
from .vector_search import vector_router, vector_search_engine

__all__ = [
    "cache_router", "similarity_router", "ttl_router", "vector_router",
    "cache_manager", "similarity_engine", "ttl_manager", "vector_search_engine"
]