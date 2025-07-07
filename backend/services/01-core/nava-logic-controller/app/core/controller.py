# app/core/controller.py
"""
NAVA Controller - Main Logic Controller
Simple and clean implementation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NAVAController:
    """
    NAVA Main Logic Controller
    Simple routing logic for AI microservices
    """
    
    def __init__(self):
        self.is_initialized = False
        self.startup_time = None
        
    async def initialize(self):
        """Initialize NAVA Controller"""
        if self.is_initialized:
            return
            
        logger.info("🚀 Initializing NAVA Controller...")
        self.is_initialized = True
        self.startup_time = datetime.now()
        logger.info("✅ NAVA Controller initialized")
        
    async def process_request(self, message: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Main chat processing method
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Simple logic test
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": f"NAVA Controller processed: {message}",
                "model_used": "nava_logic",
                "confidence": 0.8,
                "processing_time_seconds": processing_time,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing chat: {e}")
            return self._error_response(str(e))

    async def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "initialized": self.is_initialized,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "timestamp": datetime.now().isoformat()
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "response": f"Error: {error_message}",
            "model_used": "nava_error",
            "error": error_message,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }

# Global instance
nava_controller = NAVAController()

# Helper functions
async def get_controller() -> NAVAController:
    """Get controller instance"""
    return nava_controller

async def initialize_controller():
    """Initialize controller"""
    await nava_controller.initialize()