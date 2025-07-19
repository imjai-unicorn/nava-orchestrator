# File: backend/services/01-core/nava-logic-controller/app/service/microservice_client.py
"""
Microservice Client for NAVA Phase 2 Services
"""
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional
from app.config.service_discovery import service_discovery

logger = logging.getLogger(__name__)

class MicroserviceClient:
    """Client for communicating with NAVA microservices"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def call_enhanced_decision_engine(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call enhanced decision engine service"""
        try:
            endpoint = service_discovery.get_service_endpoint("decision_engine", "/analyze")
            if not endpoint:
                logger.error("Decision engine service not available")
                return None
            
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Enhanced decision engine call failed: {e}")
            return None
    
    async def call_quality_service(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call quality validation service"""
        try:
            endpoint = service_discovery.get_service_endpoint("quality_service", "/validate")
            if not endpoint:
                logger.error("Quality service not available")
                return None
            
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Quality service call failed: {e}")
            return None
    
    async def call_slf_framework(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call SLF framework service"""
        try:
            endpoint = service_discovery.get_service_endpoint("slf_framework", "/enhance")
            if not endpoint:
                logger.error("SLF framework service not available")
                return None
            
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"SLF framework call failed: {e}")
            return None
    
    async def call_cache_engine(self, action: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call cache engine service"""
        try:
            endpoint = service_discovery.get_service_endpoint("cache_engine", f"/{action}")
            if not endpoint:
                logger.error("Cache engine service not available")
                return None
            
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Cache engine call failed: {e}")
            return None
    
    async def health_check_all_services(self) -> Dict[str, bool]:
        """Check health of all Phase 2 services"""
        health_status = {}
        
        for service_name in service_discovery.services:
            health_url = service_discovery.get_health_url(service_name)
            if health_url:
                try:
                    response = await self.client.get(health_url)
                    health_status[service_name] = response.status_code == 200
                except Exception as e:
                    logger.error(f"Health check failed for {service_name}: {e}")
                    health_status[service_name] = False
            else:
                health_status[service_name] = False
        
        return health_status
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global microservice client instance
microservice_client = MicroserviceClient()