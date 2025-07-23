# main.py - Agent Registry Service
# Port: 8006
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


from app.agent_registry_service import (
    AgentRegistry, AIService, ServiceStatus, ServiceType,
    get_registry, initialize_registry, shutdown_registry,
    select_ai_service, register_local_ai_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ServiceSelectionRequest(BaseModel):
    capability: Optional[str] = None
    model_preference: Optional[str] = None
    cost_priority: bool = False
    performance_priority: bool = False

class ServiceRegistrationRequest(BaseModel):
    id: str
    name: str
    url: str
    port: int
    service_type: str  # "external_ai" or "local_ai"
    models: List[str]
    capabilities: List[str]
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 30
    priority: int = 5
    max_concurrent: int = 10

class LocalAIRegistrationRequest(BaseModel):
    service_id: str
    name: str
    port: int
    models: List[str]
    capabilities: List[str]
    max_tokens: int = 2048
    timeout_seconds: int = 10

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Agent Registry Service...")
    await initialize_registry()
    logger.info("✅ Agent Registry Service started")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Agent Registry Service...")
    await shutdown_registry()
    logger.info("✅ Agent Registry Service stopped")

# Create FastAPI app
app = FastAPI(
    title="NAVA Agent Registry",
    description="AI Service Discovery, Health Monitoring, and Load Balancing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-registry",
        "port": 8006,
        "timestamp": "2024-01-01T00:00:00Z"
    }

# Registry endpoints
@app.get("/api/services")
async def get_services(registry: AgentRegistry = Depends(get_registry)):
    """Get all registered services"""
    try:
        return {
            "services": registry.get_all_services(),
            "statistics": registry.get_statistics()
        }
    except Exception as e:
        logger.error(f"❌ Error getting services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/{service_id}")
async def get_service(service_id: str, registry: AgentRegistry = Depends(get_registry)):
    """Get specific service details"""
    try:
        service = registry.get_service_by_id(service_id)
        if not service:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
        
        return service.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting service {service_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/services/select")
async def select_service(
    request: ServiceSelectionRequest,
    registry: AgentRegistry = Depends(get_registry)
):
    """Select best available AI service"""
    try:
        service = await select_ai_service(
            capability=request.capability,
            model_preference=request.model_preference,
            cost_priority=request.cost_priority,
            performance_priority=request.performance_priority
        )
        
        if not service:
            raise HTTPException(
                status_code=503, 
                detail="No suitable AI service available"
            )
        
        # Increment load counter
        registry.increment_load(service.id)
        
        return {
            "selected_service": service.to_dict(),
            "failover_chain": [s.to_dict() for s in registry.get_failover_chain(service.id)[:3]]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error selecting service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/services/register")
async def register_service(
    request: ServiceRegistrationRequest,
    registry: AgentRegistry = Depends(get_registry)
):
    """Register a new AI service"""
    try:
        # Convert string to enum
        service_type = ServiceType.EXTERNAL_AI
        if request.service_type.lower() == "local_ai":
            service_type = ServiceType.LOCAL_AI
        
        service = AIService(
            id=request.id,
            name=request.name,
            url=request.url,
            port=request.port,
            service_type=service_type,
            models=request.models,
            capabilities=request.capabilities,
            cost_per_1k_tokens=request.cost_per_1k_tokens,
            max_tokens=request.max_tokens,
            timeout_seconds=request.timeout_seconds,
            priority=request.priority,
            max_concurrent=request.max_concurrent
        )
        
        success = registry.register_service(service)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register service")
        
        return {"message": f"Service {request.name} registered successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error registering service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/services/register-local")
async def register_local_service(
    request: LocalAIRegistrationRequest,
    registry: AgentRegistry = Depends(get_registry)
):
    """Register a local AI service (simplified)"""
    try:
        success = await register_local_ai_service(
            service_id=request.service_id,
            name=request.name,
            port=request.port,
            models=request.models,
            capabilities=request.capabilities,
            max_tokens=request.max_tokens,
            timeout_seconds=request.timeout_seconds
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register local AI service")
        
        return {"message": f"Local AI service {request.name} registered successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error registering local AI service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/services/{service_id}")
async def unregister_service(
    service_id: str,
    registry: AgentRegistry = Depends(get_registry)
):
    """Unregister an AI service"""
    try:
        success = registry.unregister_service(service_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
        
        return {"message": f"Service {service_id} unregistered successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error unregistering service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/services/{service_id}/complete")
async def complete_request(
    service_id: str,
    success: bool = True,
    registry: AgentRegistry = Depends(get_registry)
):
    """Mark request as completed (for load balancing)"""
    try:
        registry.decrement_load(service_id, success)
        return {"message": "Request completed"}
    
    except Exception as e:
        logger.error(f"❌ Error completing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health-check")
async def trigger_health_check(
    background_tasks: BackgroundTasks,
    registry: AgentRegistry = Depends(get_registry)
):
    """Trigger immediate health check"""
    try:
        background_tasks.add_task(registry._check_all_services_health)
        return {"message": "Health check triggered"}
    
    except Exception as e:
        logger.error(f"❌ Error triggering health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics(registry: AgentRegistry = Depends(get_registry)):
    """Get registry statistics"""
    try:
        return registry.get_statistics()
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/healthy-services")
async def get_healthy_services(registry: AgentRegistry = Depends(get_registry)):
    """Get only healthy services"""
    try:
        healthy_services = registry.get_healthy_services()
        return {
            "healthy_services": [service.to_dict() for service in healthy_services],
            "count": len(healthy_services)
        }
    except Exception as e:
        logger.error(f"❌ Error getting healthy services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8006))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )