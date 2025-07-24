# ==============================================================================
# Agent Registry Service - NAVA Enterprise
# ==============================================================================
# Port: 8006
# Purpose: AI Service Discovery, Health Monitoring, Load Balancing
# Created for: Local AI support and dynamic model switching
# ==============================================================================

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
import json
import os

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class ServiceType(Enum):
    """Service type enumeration"""
    EXTERNAL_AI = "external_ai"
    LOCAL_AI = "local_ai"
    HYBRID = "hybrid"

@dataclass
class AIService:
    """AI Service definition"""
    id: str
    name: str
    url: str
    port: int
    service_type: ServiceType
    models: List[str]
    capabilities: List[str]
    cost_per_1k_tokens: float
    max_tokens: int
    timeout_seconds: int
    priority: int  # 1=highest, 10=lowest
    
    # Health metrics
    status: ServiceStatus = ServiceStatus.OFFLINE
    last_health_check: Optional[datetime] = None
    response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    total_requests: int = 0
    
    # Load balancing
    current_load: int = 0
    max_concurrent: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['service_type'] = self.service_type.value
        data['status'] = self.status.value
        data['last_health_check'] = self.last_health_check.isoformat() if self.last_health_check else None
        return data

class AgentRegistry:
    """
    Agent Registry - Central AI Service Management
    """
    
    def __init__(self):
        self.services: Dict[str, AIService] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_task = None
        self.is_running = False
        
        # Performance tracking
        self.request_history = []
        self.max_history_size = 1000
        
        # Initialize default services
        self._initialize_default_services()
    
    def _initialize_default_services(self):
        """Initialize default AI services"""
        
        # External AI Services
        gpt_service = AIService(
            id="gpt-client",
            name="OpenAI GPT",
            url=os.getenv("GPT_SERVICE_URL", "https://nava-orchestrator-gpt-production.up.railway.app"),
            port=8002,
            service_type=ServiceType.EXTERNAL_AI,
            models=["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            capabilities=["chat", "completion", "reasoning", "coding"],
            cost_per_1k_tokens=0.002,
            max_tokens=4096,
            timeout_seconds=30,
            priority=2
        )
        
        claude_service = AIService(
            id="claude-client", 
            name="Anthropic Claude",
            url=os.getenv("CLAUDE_SERVICE_URL", "https://nava-orchestrator-claude-production.up.railway.app"),
            port=8003,
            service_type=ServiceType.EXTERNAL_AI,
            models=["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
            capabilities=["chat", "reasoning", "analysis", "writing"],
            cost_per_1k_tokens=0.0015,
            max_tokens=4096,
            timeout_seconds=45,
            priority=1  # Highest priority for reasoning
        )
        
        gemini_service = AIService(
            id="gemini-client",
            name="Google Gemini",
            url=os.getenv("GEMINI_SERVICE_URL", "https://nava-orchestrator-gemini-production.up.railway.app"), 
            port=8004,
            service_type=ServiceType.EXTERNAL_AI,
            models=["gemini-2.0-flash-exp", "gemini-2.5-pro"],
            capabilities=["chat", "multimodal", "search", "reasoning"],
            cost_per_1k_tokens=0.001,
            max_tokens=8192,
            timeout_seconds=25,
            priority=3
        )
        
        # Register services
        self.services[gpt_service.id] = gpt_service
        self.services[claude_service.id] = claude_service
        self.services[gemini_service.id] = gemini_service
        
        logger.info("✅ Default AI services initialized")
    
    async def start(self):
        """Start the agent registry"""
        if self.is_running:
            return
            
        logger.info("🚀 Starting Agent Registry...")
        self.is_running = True
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Initial health check
        await self._check_all_services_health()
        
        logger.info("✅ Agent Registry started")
    
    async def stop(self):
        """Stop the agent registry"""
        if not self.is_running:
            return
            
        logger.info("🛑 Stopping Agent Registry...")
        self.is_running = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Agent Registry stopped")
    
    async def _health_check_loop(self):
        """Continuous health checking loop"""
        while self.is_running:
            try:
                await self._check_all_services_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _check_all_services_health(self):
        """Check health of all registered services"""
        tasks = []
        for service in self.services.values():
            task = asyncio.create_task(self._check_service_health(service))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: AIService):
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"{service.url}/health"
                
                async with session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        service.status = ServiceStatus.HEALTHY
                        service.response_time_ms = response_time
                        service.last_health_check = datetime.now()
                        logger.debug(f"✅ {service.name} healthy ({response_time:.1f}ms)")
                    else:
                        service.status = ServiceStatus.DEGRADED
                        service.error_count += 1
                        logger.warning(f"⚠️ {service.name} degraded (HTTP {response.status})")
                        
        except asyncio.TimeoutError:
            service.status = ServiceStatus.UNHEALTHY
            service.error_count += 1
            logger.warning(f"⏰ {service.name} timeout")
            
        except Exception as e:
            service.status = ServiceStatus.OFFLINE
            service.error_count += 1
            logger.error(f"❌ {service.name} offline: {e}")
    
    def register_service(self, service: AIService) -> bool:
        """Register a new AI service"""
        try:
            self.services[service.id] = service
            logger.info(f"✅ Registered service: {service.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register service {service.name}: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister an AI service"""
        try:
            if service_id in self.services:
                service_name = self.services[service_id].name
                del self.services[service_id]
                logger.info(f"✅ Unregistered service: {service_name}")
                return True
            else:
                logger.warning(f"⚠️ Service not found: {service_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to unregister service {service_id}: {e}")
            return False
    
    def get_best_service(self, 
                        capability: Optional[str] = None,
                        model_preference: Optional[str] = None,
                        cost_priority: bool = False,
                        performance_priority: bool = False) -> Optional[AIService]:
        """
        Get the best available service based on criteria
        """
        available_services = [
            s for s in self.services.values() 
            if s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            and s.current_load < s.max_concurrent
        ]
        
        if not available_services:
            logger.warning("⚠️ No healthy services available")
            return None
        
        # Filter by capability
        if capability:
            available_services = [
                s for s in available_services 
                if capability in s.capabilities
            ]
        
        # Filter by model preference
        if model_preference:
            preferred_services = [
                s for s in available_services 
                if model_preference in s.models
            ]
            if preferred_services:
                available_services = preferred_services
        
        if not available_services:
            logger.warning(f"⚠️ No services available for capability: {capability}")
            return None
        
        # Sort by criteria
        if cost_priority:
            # Sort by cost (lowest first)
            available_services.sort(key=lambda s: s.cost_per_1k_tokens)
        elif performance_priority:
            # Sort by response time and success rate
            available_services.sort(key=lambda s: (s.response_time_ms, -s.success_rate))
        else:
            # Sort by priority and health
            available_services.sort(key=lambda s: (
                s.priority,
                s.current_load / s.max_concurrent,
                s.response_time_ms
            ))
        
        selected_service = available_services[0]
        logger.info(f"🎯 Selected service: {selected_service.name}")
        return selected_service
    
    def get_failover_chain(self, primary_service_id: str) -> List[AIService]:
        """Get failover chain for a service"""
        chain = []
        
        # Find all healthy services except primary
        available_services = [
            s for s in self.services.values()
            if s.id != primary_service_id 
            and s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            and s.current_load < s.max_concurrent
        ]
        
        # Sort by priority
        available_services.sort(key=lambda s: s.priority)
        chain.extend(available_services)
        
        return chain
    
    def increment_load(self, service_id: str):
        """Increment service load counter"""
        if service_id in self.services:
            self.services[service_id].current_load += 1
            self.services[service_id].total_requests += 1
    
    def decrement_load(self, service_id: str, success: bool = True):
        """Decrement service load counter and update metrics"""
        if service_id in self.services:
            service = self.services[service_id]
            service.current_load = max(0, service.current_load - 1)
            
            if not success:
                service.error_count += 1
            
            # Update success rate
            if service.total_requests > 0:
                service.success_rate = 1.0 - (service.error_count / service.total_requests)
    
    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered services"""
        return {
            service_id: service.to_dict() 
            for service_id, service in self.services.items()
        }
    
    def get_service_by_id(self, service_id: str) -> Optional[AIService]:
        """Get service by ID"""
        return self.services.get(service_id)
    
    def get_healthy_services(self) -> List[AIService]:
        """Get all healthy services"""
        return [
            s for s in self.services.values()
            if s.status == ServiceStatus.HEALTHY
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_services = len(self.services)
        healthy_services = len(self.get_healthy_services())
        
        total_requests = sum(s.total_requests for s in self.services.values())
        total_errors = sum(s.error_count for s in self.services.values())
        
        avg_response_time = 0.0
        if healthy_services > 0:
            avg_response_time = sum(
                s.response_time_ms for s in self.services.values()
                if s.status == ServiceStatus.HEALTHY
            ) / healthy_services
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "average_response_time_ms": avg_response_time,
            "registry_uptime": datetime.now().isoformat(),
            "services_status": {
                service.id: {
                    "name": service.name,
                    "status": service.status.value,
                    "load": f"{service.current_load}/{service.max_concurrent}",
                    "success_rate": f"{service.success_rate:.2%}"
                }
                for service in self.services.values()
            }
        }

# Global registry instance
agent_registry = AgentRegistry()

# Helper functions for FastAPI
async def get_registry() -> AgentRegistry:
    """Get registry instance"""
    return agent_registry

async def initialize_registry():
    """Initialize and start registry"""
    await agent_registry.start()

async def shutdown_registry():
    """Shutdown registry"""
    await agent_registry.stop()

# Service selection helpers
async def select_ai_service(
    capability: Optional[str] = None,
    model_preference: Optional[str] = None,
    cost_priority: bool = False,
    performance_priority: bool = False
) -> Optional[AIService]:
    """Helper function to select best AI service"""
    return agent_registry.get_best_service(
        capability=capability,
        model_preference=model_preference,
        cost_priority=cost_priority,
        performance_priority=performance_priority
    )

async def register_local_ai_service(
    service_id: str,
    name: str,
    port: int,
    models: List[str],
    capabilities: List[str],
    max_tokens: int = 2048,
    timeout_seconds: int = 10
) -> bool:
    """Helper to register local AI services"""
    
    local_service = AIService(
        id=service_id,
        name=name,
        url=f"http://localhost:{port}", 
        port=port,
        service_type=ServiceType.LOCAL_AI,
        models=models,
        capabilities=capabilities,
        cost_per_1k_tokens=0.0,  # Local = free
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        priority=1,  # Local AI gets highest priority
        max_concurrent=20  # Local can handle more concurrent
    )
    
    return agent_registry.register_service(local_service)