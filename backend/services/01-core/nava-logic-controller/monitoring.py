# monitoring.py - NO PSUTIL VERSION
import time
from datetime import datetime
from typing import Dict, Any
import logging
from fastapi import APIRouter
import sys # Import sys
import os  # Import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
# current_file_dir -> E:\nava-projects\backend\services\01-core\nava-logic-controller

# Path to 'app' directory (e.g., E:\nava-projects\backend\services\01-core\nava-logic-controller\app)
app_dir = os.path.join(current_file_dir, 'app')

if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

print(f"🔍 Current monitoring file dir: {current_file_dir}")
print(f"🔍 Adding app_dir to sys.path: {app_dir}")
# --- END PATH SETUP ---

# เปลี่ยนจาก relative import เป็น absolute import
from app.service.logic_orchestrator import LogicOrchestrator
router = APIRouter()
# Initializing orchestrator globally might have implications for state management
# in a production FastAPI app. Consider using dependency injection.
orchestrator = LogicOrchestrator()

logger = logging.getLogger(__name__)

@router.get("/health")
async def get_system_health():
    """Get current system health and stabilization status"""
    # ensure orchestrator is initialized if needed
    if not orchestrator.is_initialized:
        await orchestrator.initialize() # Call initialize if not already
    return orchestrator.get_system_status()

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache performance statistics"""
    if orchestrator.cache_manager: # Check if cache_manager is initialized
        return orchestrator.cache_manager.get_cache_stats()
    return {"status": "Cache Manager not available"}

@router.get("/features/status")
async def get_feature_status():
    """Get feature activation status"""
    if orchestrator.feature_manager: # Check if feature_manager is initialized
        return orchestrator.feature_manager.get_feature_status()
    return {"status": "Feature Manager not available"}

@router.post("/features/{feature_name}/enable")
async def enable_feature(feature_name: str):
    """Force enable a feature (admin only)"""
    if orchestrator.feature_manager: # Check if feature_manager is initialized
        # In feature_flags.py, force_enable is a method of ProgressiveFeatureManager
        # And there's a global function force_enable_feature that calls it.
        # So either call orchestrator.feature_manager.force_enable(feature_name)
        # or import and use the global force_enable_feature directly.
        # Based on monitoring.py, it expects a method on feature_manager directly.
        success = orchestrator.feature_manager.force_enable(feature_name) # Corrected to .force_enable()
        return {"success": success, "feature": feature_name, "status": "enabled"}
    return {"success": False, "message": "Feature Manager not available"}

@router.post("/features/{feature_name}/disable")
async def disable_feature(feature_name: str):
    """Force disable a feature (admin only)"""
    if orchestrator.feature_manager: # Check if feature_manager is initialized
        success = orchestrator.feature_manager.force_disable(feature_name) # Corrected to .force_disable()
        return {"success": success, "feature": feature_name, "status": "disabled"}
    return {"success": False, "message": "Feature Manager not available"}

@router.post("/cache/clear")
async def clear_cache():
    """Clear response cache (admin only)"""
    if orchestrator.cache_manager: # Check if cache_manager is initialized
        orchestrator.cache_manager.clear_cache()
        return {"success": True, "message": "Cache cleared"}
    return {"success": False, "message": "Cache Manager not available"}

class PerformanceMonitor:
    """Simple performance monitoring for NAVA - NO PSUTIL"""
    
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "response_times": [],
            "ai_model_usage": {},
            "error_count": 0,
            "start_time": datetime.now()
        }
        logger.info("✅ Performance Monitor initialized (no psutil)")
    
    def track_request(self, endpoint: str, response_time: float, ai_model: str = None):
        """Track each request performance"""
        try:
            self.metrics["request_count"] += 1
            self.metrics["response_times"].append(response_time)
            
            # Keep only last 100 records
            if len(self.metrics["response_times"]) > 100:
                self.metrics["response_times"] = self.metrics["response_times"][-100:]
            
            # Track AI model usage
            if ai_model:
                if ai_model not in self.metrics["ai_model_usage"]:
                    self.metrics["ai_model_usage"][ai_model] = 0
                self.metrics["ai_model_usage"][ai_model] += 1
                
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
    
    def track_error(self, error_type: str, endpoint: str = None):
        """Track errors"""
        try:
            self.metrics["error_count"] += 1
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics - NO SYSTEM STATS"""
        try:
            # Calculate averages
            avg_response_time = 0
            if self.metrics["response_times"]:
                avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.metrics["start_time"]).total_seconds()
            
            return {
                "status": "healthy",
                "service": "NAVA Logic Controller",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "total_requests": self.metrics["request_count"],
                "average_response_time_ms": round(avg_response_time * 1000, 2),
                "error_count": self.metrics["error_count"],
                "ai_model_usage": self.metrics["ai_model_usage"],
                "uptime_hours": round(uptime_seconds / 3600, 2)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get simple health summary"""
        try:
            metrics = self.get_metrics()
            error_rate = (metrics["error_count"] / max(1, metrics["total_requests"])) * 100
            
            return {
                "status": "healthy" if error_rate < 10 else "degraded",
                "total_requests": metrics["total_requests"],
                "error_rate": f"{error_rate:.2f}%",
                "uptime": f"{metrics['uptime_hours']:.1f} hours",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global monitor instance
monitor = PerformanceMonitor()