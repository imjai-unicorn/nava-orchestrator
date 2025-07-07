# app/core/__init__.py - Updated for Enhanced Components
"""NAVA Core Module - Enhanced Version"""

# Import enhanced components
from .core.controller import NAVAController, get_controller, initialize_controller
from .core.decision_engine import EnhancedDecisionEngine, BehaviorType, WorkflowMode

__all__ = [
    'NAVAController', 
    'get_controller', 
    'initialize_controller',
    'EnhancedDecisionEngine',
    'BehaviorType',
    'WorkflowMode'
]

# Version info
__version__ = "2.0.0"
__description__ = "Enhanced NAVA Logic Controller with Behavior-First AI Selection"