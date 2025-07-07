# monitoring.py - Simple Performance Monitor
import time
import psutil
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Simple performance monitoring for NAVA"""
    
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "response_times": [],
            "ai_model_usage": {},
            "error_count": 0,
            "start_time": datetime.now(),
            "endpoint_stats": {}
        }
        logger.info("✅ Performance Monitor initialized")
    
    def track_request(self, endpoint: str, response_time: float, ai_model: str = None):
        """Track each request performance"""
        try:
            self.metrics["request_count"] += 1
            self.metrics["response_times"].append(response_time)
            
            # Keep only last 1000 records
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
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
            logger.warning(f"Error tracked: {error_type} on {endpoint}")
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            # Calculate averages
            avg_response_time = 0
            if self.metrics["response_times"]:
                avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.metrics["start_time"]).total_seconds()
            
            # Get system stats safely
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
            except:
                memory_mb = 0
                cpu_percent = 0
            
            return {
                "status": "healthy",
                "service": "NAVA Logic Controller",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "total_requests": self.metrics["request_count"],
                "average_response_time_ms": round(avg_response_time * 1000, 2),
                "error_count": self.metrics["error_count"],
                "ai_model_usage": self.metrics["ai_model_usage"],
                "uptime_hours": round(uptime_seconds / 3600, 2),
                "memory_usage_mb": round(memory_mb, 2),
                "cpu_percent": cpu_percent
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
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
                "memory_mb": metrics["memory_usage_mb"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global monitor instance
monitor = PerformanceMonitor()"# test change" 
