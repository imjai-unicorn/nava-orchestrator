# backend/services/05-enhanced-intelligence/decision-engine/app/dependencies.py
"""Decision Engine Dependencies"""

from fastapi import Depends, HTTPException, Header
from typing import Optional, Dict, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

async def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Validate API key if provided"""
    if x_api_key:
        expected_key = os.getenv("DECISION_API_KEY")
        if expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def get_decision_config() -> Dict[str, Any]:
    """Get decision engine configuration"""
    return {
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.8")),
        "max_alternatives": int(os.getenv("MAX_ALTERNATIVES", "3")),
        "decision_timeout": int(os.getenv("DECISION_TIMEOUT", "10"))
    }

async def get_decision_engine():
    """Get decision engine instance"""
    from .decision_engine import decision_engine
    return decision_engine

