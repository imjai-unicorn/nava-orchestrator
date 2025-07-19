# File: backend/services/01-core/nava-logic-controller/app/config/service_discovery.py
"""
Service Discovery Configuration for NAVA Microservices
"""
import os
from typing import Dict, Optional

class ServiceDiscovery:
    """Centralized service discovery configuration"""
    
    def __init__(self):
        self.services = {
            "decision_engine": {
                "url": os.getenv("DECISION_ENGINE_URL", "http://localhost:8008"),
                "health_path": "/health",
                "api_prefix": "/api/decision"
            },
            "quality_service": {
                "url": os.getenv("QUALITY_SERVICE_URL", "http://localhost:8009"),
                "health_path": "/health",
                "api_prefix": "/api/quality"
            },
            "slf_framework": {
                "url": os.getenv("SLF_FRAMEWORK_URL", "http://localhost:8010"),
                "health_path": "/health",
                "api_prefix": "/api/slf"
            },
            "cache_engine": {
                "url": os.getenv("CACHE_ENGINE_URL", "http://localhost:8013"),
                "health_path": "/health",
                "api_prefix": "/api/cache"
            }
        }
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get service URL by name"""
        service = self.services.get(service_name)
        return service["url"] if service else None
    
    def get_service_endpoint(self, service_name: str, endpoint: str) -> Optional[str]:
        """Get full endpoint URL for a service"""
        service = self.services.get(service_name)
        if service:
            return f"{service['url']}{service['api_prefix']}{endpoint}"
        return None
    
    def get_health_url(self, service_name: str) -> Optional[str]:
        """Get health check URL for a service"""
        service = self.services.get(service_name)
        if service:
            return f"{service['url']}{service['health_path']}"
        return None

# Global service discovery instance
service_discovery = ServiceDiscovery()