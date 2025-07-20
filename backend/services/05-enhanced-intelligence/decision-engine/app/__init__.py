# ===== 1. DECISION ENGINE SERVICE (Port 8008) =====

# backend/services/05-enhanced-intelligence/decision-engine/app/__init__.py
"""
Decision Engine Service Package
AI model selection and decision optimization system
"""

__version__ = "1.0.0"
__service__ = "decision-engine"

from .enhanced_decision_engine import decision_engine, enhanced_decision_router
from .criteria_analyzer import criteria_router, criteria_analyzer
from .outcome_predictor import outcome_router, outcome_predictor
from .risk_assessor import risk_router, risk_assessor

__all__ = [
    "enhanced_decision_router", "criteria_router", "outcome_router", "risk_router",
    "decision_engine", "criteria_analyzer", "outcome_predictor", "risk_assessor"
] 
