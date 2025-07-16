# backend/services/05-enhanced-intelligence/quality-service/app/__init__.py
"""
Quality Service Package
Multi-dimensional response quality validation system
"""

__version__ = "1.0.0"
__service__ = "quality-service"

# Package exports
from .quality_validator import quality_router, quality_validator
from .response_scorer import scorer_router, response_scorer  
from .improvement_suggester import improvement_router, improvement_suggester

__all__ = [
    "quality_router",
    "scorer_router", 
    "improvement_router",
    "quality_validator",
    "response_scorer",
    "improvement_suggester"
]
