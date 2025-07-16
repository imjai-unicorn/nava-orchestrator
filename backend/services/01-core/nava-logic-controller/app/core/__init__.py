# app/core/__init__.py
"""NAVA Core Module"""

# Import from controller.py
from .controller import NAVAController, get_controller, initialize_controller

# Import from decision_engine.py  
from .decision_engine import EnhancedDecisionEngine as DecisionEngine
try:
    from .feature_flags import (
        ProgressiveFeatureManager,
        feature_manager,
        is_feature_enabled,
        record_feature_usage,
        get_feature_status,
        force_enable_feature
    )
    FEATURE_FLAGS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Feature flags not available: {e}")
    FEATURE_FLAGS_AVAILABLE = False

# Make available at package level
__all__ = [
    'NAVAController', 
    'get_controller', 
    'initialize_controller',
    'DecisionEngine',
    # ✅ เพิ่ม feature flags
    'ProgressiveFeatureManager',
    'feature_manager',
    'is_feature_enabled',
    'record_feature_usage', 
    'get_feature_status',
    'force_enable_feature',
    'FEATURE_FLAGS_AVAILABLE'
]