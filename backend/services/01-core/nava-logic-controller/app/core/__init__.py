# app/core/__init__.py
"""NAVA Core Module"""

# Import from controller.py
from .controller import NAVAController, get_controller, initialize_controller

# Import from decision_engine.py  
from .decision_engine import EnhancedDecisionEngine as DecisionEngine
# Make available at package level
__all__ = [
    'NAVAController', 
    'get_controller', 
    'initialize_controller',
    'DecisionEngine'
]