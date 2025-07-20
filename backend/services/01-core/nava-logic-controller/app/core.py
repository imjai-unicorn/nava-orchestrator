# app/core.py
"""
NAVA Logic Controller - Main Logic System
Pure logic controller for intelligent microservice routing
"""
import os
import httpx
import logging
from typing import Dict, Any, Optional
from app.core.core import CoreSystem
from datetime import datetime

class CoreSystem:
    def __init__(self):
        self.status = "ready"
        self.version = "1.0.0"

    def run_diagnostics(self):
        return {
            "core_status": self.status,
            "version": self.version,
            "timestamp": datetime.now().isoformat()
        }


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAVALogicController:
    """
    NAVA Logic Controller - Main Logic System
    Pure logic system for routing user requests to appropriate AI microservices
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.is_initialized = False
        self.startup_time = None
        
    async def initialize(self):
        """
        Initialize NAVA logic system
        """
        if self.is_initialized:
            return
            
        self.logger.info("🚀 Initializing NAVA Logic Controller...")
        self.is_initialized = True
        self.startup_time = datetime.now()
        self.logger.info("✅ NAVA Logic Controller initialized successfully")
        
    async def process_request(self, message: str, user_id: str = "anonymous", 
                          user_preference: Optional[str] = None,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main chat processing method - uses logic to route to best microservice
        """
        
        # Ensure system is initialized
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🎯 Processing chat request from user: {user_id}")
            
            # Validate input
            if not message or not message.strip():
                return self._error_response("Message cannot be empty")
            
            # Simple test response for now
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": f"Logic Controller processed: {message}",
                "model_used": "logic_test",
                "confidence": 0.8,
                "processing_time_seconds": processing_time,
                "user_id": user_id,
                "nava_logic_version": "1.0.0",
                "controller_type": "logic_system",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing chat: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": "ขออภัย เกิดข้อผิดพลาดในระบบ logic controller กรุณาลองใหม่อีกครั้ง",
                "model_used": "logic_error",
                "error": str(e),
                "confidence": 0.0,
                "processing_time_seconds": processing_time,
                "user_id": user_id,
                "controller_type": "logic_system",
                "timestamp": datetime.now().isoformat()
            }

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of logic controller
        """
        
        health_data = {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "logic_controller_initialized": self.is_initialized,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            "controller_type": "logic_system",
            "timestamp": datetime.now().isoformat()
        }
        
        return health_data

    async def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available microservices
        """
        
        if not self.is_initialized:
            return {
                "available_models": [],
                "error": "Logic controller not initialized",
                "controller_type": "logic_system",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "available_models": ["gpt", "claude", "gemini"],  # Mock for testing
            "total_services": 3,
            "healthy_services": 0,  # No real services connected yet
            "controller_type": "logic_system",
            "timestamp": datetime.now().isoformat()
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response
        """
        return {
            "response": f"ข้อผิดพลาด: {error_message}",
            "model_used": "logic_error",
            "error": error_message,
            "confidence": 0.0,
            "controller_type": "logic_system",
            "timestamp": datetime.now().isoformat()
        }

# Global instance for FastAPI
nava_logic_controller = NAVALogicController()


# Helper functions for FastAPI endpoints
async def get_nava_controller() -> NAVALogicController:
    """
    Get NAVA logic controller instance (for dependency injection)
    """
    return nava_logic_controller

async def initialize_nava():
    """
    Initialize NAVA logic system (call during FastAPI startup)
    """
    await nava_logic_controller.initialize()

async def process_request_request(message: str, user_id: str = "anonymous", 
                             user_preference: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process chat request through NAVA logic controller
    """
    return await nava_logic_controller.process_request(message, user_id, user_preference, context)

# All possible variations
NAVAlogicController = NAVALogicController
NAVALogiccontroller = NAVALogicController
NAVAlogiccontroller = NAVALogicController
navalogiccontroller = NAVALogicController