# ===== 2. SLF FRAMEWORK SERVICE (Port 8010) =====

# backend/services/05-enhanced-intelligence/slf-framework/app/__init__.py
"""
SLF Framework Service Package
Systematic reasoning and cognitive enhancement system
"""

__version__ = "1.0.0"
__service__ = "slf-framework"

from .slf_enhancer import slf_router, slf_enhancer
from .cognitive_framework import cognitive_router, cognitive_framework
from .reasoning_validator import reasoning_router, reasoning_validator

__all__ = [
    "slf_router", "cognitive_router", "reasoning_router",
    "slf_enhancer", "cognitive_framework", "reasoning_validator"
]