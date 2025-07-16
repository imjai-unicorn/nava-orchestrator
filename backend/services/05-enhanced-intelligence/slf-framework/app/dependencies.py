# backend/services/05-enhanced-intelligence/slf-framework/app/dependencies.py
"""SLF Framework Dependencies"""

from fastapi import Depends, HTTPException, Header
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

async def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Validate API key if provided"""
    if x_api_key:
        expected_key = os.getenv("SLF_API_KEY")
        if expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def get_slf_config() -> Dict[str, Any]:
    """Get SLF framework configuration"""
    return {
        "enhancement_level": os.getenv("ENHANCEMENT_LEVEL", "standard"),
        "reasoning_depth": int(os.getenv("REASONING_DEPTH", "3")),
        "max_prompt_length": int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
    }

async def get_slf_enhancer():
    """Get SLF enhancer instance"""
    from .slf_enhancer import slf_enhancer
    return slf_enhancer
