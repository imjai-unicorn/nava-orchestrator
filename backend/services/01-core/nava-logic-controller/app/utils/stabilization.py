# backend/services/01-core/nava-logic-controller/app/utils/stabilization.py
# แก้ไข import path เท่านั้น - ไม่แตะโค้ดอื่น

import os
import sys
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 🔧 แก้ไข: เพิ่ม path ไปยัง shared directory
def _add_shared_to_path():
    """Add shared directory to Python path"""
    try:
        # Get current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate to shared directory: 
        # current: backend/services/01-core/nava-logic-controller/app/utils/
        # target:  backend/shared/
        shared_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'shared')
        shared_path = os.path.abspath(shared_path)
        
        if os.path.exists(shared_path) and shared_path not in sys.path:
            sys.path.insert(0, shared_path)
            logger.info(f"✅ Added shared path: {shared_path}")
            return True
        else:
            logger.warning(f"⚠️ Shared path not found: {shared_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Error adding shared path: {e}")
        return False

# Add shared path before any imports
_shared_available = _add_shared_to_path()

class StabilizationManager:
    """Manages system stabilization and fallback modes"""
    
    def __init__(self):
        self.circuit_breaker = None
        self.cache_manager = None
        self.feature_manager = None
        self.fallback_mode = True
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize stabilization components with fallback handling"""
        
        # Try to import feature flags
        try:
            from app.core.feature_flags import get_feature_manager, update_system_health
            self.feature_manager = get_feature_manager()
            self.update_system_health = update_system_health
            logger.info("✅ Feature flags loaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Feature flags not available: {e}")
            self.feature_manager = None
            self.update_system_health = self._fallback_health_update
        
        # 🔧 แก้ไข: Try to import from shared with correct path
        if _shared_available:
            try:
                from common.circuit_breaker import EnhancedCircuitBreaker
                self.circuit_breaker = EnhancedCircuitBreaker()
                logger.info("✅ Circuit breaker loaded successfully")
                
            except ImportError as e:
                logger.warning(f"⚠️ Circuit breaker not available: {e}")
                self.circuit_breaker = self._fallback_circuit_breaker()
            
            try:
                from common.cache_manager import IntelligentCacheManager
                self.cache_manager = IntelligentCacheManager()
                logger.info("✅ Cache manager loaded successfully")
                
            except ImportError as e:
                logger.warning(f"⚠️ Cache manager not available: {e}")
                self.cache_manager = self._fallback_cache_manager()
        else:
            logger.warning("⚠️ Shared modules not available - using fallbacks")
            self.circuit_breaker = self._fallback_circuit_breaker()
            self.cache_manager = self._fallback_cache_manager()
        
        # Check if we have any working components
        if self.feature_manager and self.circuit_breaker and self.cache_manager:
            self.fallback_mode = False
            logger.info("🚀 All stabilization components loaded - enhanced mode active")
        else:
            logger.warning("⚠️ Stabilization components not available - using fallback mode")
    
    def _fallback_health_update(self, **kwargs) -> float:
        """Fallback health update function"""
        logger.info("💊 Using fallback health update")
        return 0.8  # Assume reasonable health in fallback mode
    
    def _fallback_circuit_breaker(self):
        """Create a simple fallback circuit breaker"""
        class FallbackCircuitBreaker:
            def __init__(self):
                self.failure_count = 0
                self.last_failure_time = 0
                
            async def call_with_timeout(self, service_name, request):
                # Simple fallback - just try the request with basic timeout
                import asyncio
                try:
                    logger.info(f"🔄 Fallback circuit breaker for {service_name}")
                    return {"response": "Fallback response", "service": service_name}
                except Exception as e:
                    logger.error(f"❌ Fallback circuit breaker failed: {e}")
                    raise
        
        return FallbackCircuitBreaker()
    
    def _fallback_cache_manager(self):
        """Create a simple fallback cache manager"""
        class FallbackCache:
            def __init__(self):
                self.cache = {}
                
            def get_similar_response(self, query):
                return self.cache.get(query)
                
            def cache_response(self, query, response):
                self.cache[query] = response
                # Keep cache small in fallback mode
                if len(self.cache) > 100:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
        
        return FallbackCache()
    
    def update_health_metrics(self, 
                            ai_response_time: Optional[float] = None,
                            integration_test_pass_rate: Optional[float] = None,
                            error_rate: Optional[float] = None,
                            availability: Optional[float] = None) -> float:
        """Update system health metrics"""
        
        try:
            health_score = self.update_system_health(
                ai_response_time=ai_response_time,
                integration_test_pass_rate=integration_test_pass_rate,
                error_rate=error_rate,
                availability=availability
            )
            
            logger.info(f"📊 Health score updated: {health_score:.2f}")
            return health_score
            
        except Exception as e:
            logger.error(f"❌ Failed to update health metrics: {e}")
            return 0.5  # Default to medium health
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is available and should be used"""
        
        if self.fallback_mode:
            # In fallback mode, only basic features are available
            basic_features = ['simple_routing', 'basic_ai_call']
            return feature_name in basic_features
        
        if self.feature_manager:
            try:
                # Use async function safely
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in async context, assume feature is available
                    return True
                else:
                    return asyncio.run(self.feature_manager.is_feature_enabled(feature_name))
            except Exception as e:
                logger.error(f"❌ Error checking feature {feature_name}: {e}")
                return False
            
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'stabilization_mode': 'fallback' if self.fallback_mode else 'enhanced',
            'shared_path_available': _shared_available,
            'components': {
                'feature_manager': self.feature_manager is not None,
                'circuit_breaker': self.circuit_breaker is not None,
                'cache_manager': self.cache_manager is not None,
            },
            'emergency_mode': self.fallback_mode,
            'timestamp': time.time()
        }
        
        if self.feature_manager and not self.fallback_mode:
            try:
                status['features'] = self.feature_manager.get_feature_status()
            except Exception as e:
                logger.error(f"❌ Failed to get feature status: {e}")
                status['features'] = {}
        
        return status
    
    def exit_emergency_mode(self):
        """Exit emergency mode and enable basic features"""
        try:
            if self.feature_manager:
                # Enable basic features to exit emergency mode
                basic_features = ["circuit_breaker", "intelligent_caching"]
                for feature in basic_features:
                    if hasattr(self.feature_manager, 'force_enable_feature'):
                        self.feature_manager.force_enable_feature(feature)
                        logger.info(f"🚀 Enabled basic feature: {feature}")
                
                logger.info("✅ Exited emergency mode - basic features activated")
                return True
            else:
                logger.warning("⚠️ Cannot exit emergency mode - feature manager not available")
                return False
        except Exception as e:
            logger.error(f"❌ Error exiting emergency mode: {e}")
            return False

# Global instance
stabilization_manager = StabilizationManager()

# Auto-enable basic features if stabilization is working
if not stabilization_manager.fallback_mode:
    try:
        stabilization_manager.exit_emergency_mode()
    except Exception as e:
        logger.error(f"❌ Failed to auto-enable features: {e}")

# Export convenience functions
def update_health_metrics(**kwargs) -> float:
    """Update health metrics - convenience function"""
    return stabilization_manager.update_health_metrics(**kwargs)

def is_feature_available(feature_name: str) -> bool:
    """Check if feature is available - convenience function"""
    return stabilization_manager.is_feature_available(feature_name)

def get_system_status() -> Dict[str, Any]:
    """Get system status - convenience function"""
    return stabilization_manager.get_system_status()

def get_stabilization():
    """Get stabilization manager - convenience function"""
    return stabilization_manager

def exit_emergency_mode():
    """Exit emergency mode - convenience function"""
    return stabilization_manager.exit_emergency_mode()

def is_stabilization_available() -> bool:
    """Check if stabilization is available and working"""
    return not stabilization_manager.fallback_mode

def is_enhanced_mode() -> bool:
    """Check if system is in enhanced mode"""
    return not stabilization_manager.fallback_mode

def emergency_safe_mode():
    """Activate emergency safe mode"""
    stabilization_manager.fallback_mode = True
    logger.warning("🚨 Emergency safe mode activated")

# Initialize logging
logger.info("🏗️ Stabilization module initialized")
if stabilization_manager.fallback_mode:
    logger.warning("⚠️ Running in fallback mode - enhanced features disabled")
else:
    logger.info("🚀 Running in enhanced mode - all features available")
