# app/service/service_discovery.py
"""
Simple Service Discovery for NAVA
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ServiceDiscovery:
    """
    Simple Service Discovery
    """
    
    def __init__(self):
        self.services = {
            "gpt": {"url": "http://localhost:8002", "status": "unknown"},
            "claude": {"url": "http://localhost:8003", "status": "unknown"}, 
            "gemini": {"url": "http://localhost:8004", "status": "unknown"}
        }
        
    def get_available_models(self) -> List[str]:
        """Get available models"""
        return ["gpt", "claude", "gemini"]  # Mock for testing
        
    def is_service_available(self, model: str) -> bool:
        """Check if service is available"""
        return model in self.services
        
    def get_service_url(self, model: str, endpoint: str = "chat") -> str:
        """Get service URL"""
        if model in self.services:
            return f"{self.services[model]['url']}/{endpoint}"
        return ""
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return self.services
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        return {
            "total_services": len(self.services),
            "available_services": len(self.services),
            "status": "mock"
        }
        
    async def start_monitoring(self):
        """Start monitoring"""
        logger.info("Service monitoring started (mock)")
        
    async def check_all_services(self) -> Dict[str, str]:
        """Check all services"""
        return {name: "mock" for name in self.services.keys()}