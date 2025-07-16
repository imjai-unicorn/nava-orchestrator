# backend/services/05-enhanced-intelligence/cache-engine/app/dependencies.py
"""Cache Engine Dependencies"""

from fastapi import Depends, HTTPException, Header
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

async def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Validate API key if provided"""
    if x_api_key:
        expected_key = os.getenv("CACHE_API_KEY")
        if expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def get_cache_config() -> Dict[str, Any]:
    """Get cache engine configuration"""
    return {
        "memory_max_size": int(os.getenv("MEMORY_MAX_SIZE", "1000")),
        "default_ttl": int(os.getenv("DEFAULT_TTL", "3600")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    }

async def get_cache_manager():
    """Get cache manager instance"""
    from .cache_manager import cache_manager
    return cache_manager