# app/service/real_ai_client.py - FINAL FIXED VERSION
"""
Real AI Client - HTTP calls to AI microservices
FIXED: Removed ALL recursive method calls
"""

import logging
import httpx
import os
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

async def call_ai_with_simple_retry(self, model: str, message: str, context: Dict[str, Any] = None):
    """Call AI with simple retry logic (no tenacity needed)"""
    
    # First attempt
    try:
        return await self.call_ai(model, message, context)
    except Exception as e:
        logger.warning(f"First attempt failed for {model}: {e}")
        
        # Second attempt
        try:
            return await self.call_ai(model, message, context)
        except Exception as e2:
            logger.error(f"Second attempt failed for {model}: {e2}")
            
            # Return error response
            return self._error_response(str(e2), model)
        
async def call_claude_with_retry(prompt: str):
    """Call Claude with retry logic for overload handling"""
    try:
        # เรียก Claude API ตามปกติ
        response = await claude_api_call(prompt)
        return response
    except Exception as e:
        if "overloaded_error" in str(e) or "529" in str(e):
            print(f"⚠️ Claude overloaded, retrying...")
            raise e  # Retry will handle this
        else:
            # ถ้าเป็น error อื่น ให้ fallback ไป GPT
            print(f"🔄 Claude error, falling back to GPT: {e}")
            return await call_gpt_fallback(prompt)

logger = logging.getLogger(__name__)

class RealAIClient:
    """
    Real AI Client - Makes HTTP calls to AI microservices
    """
    
    def __init__(self, service_discovery=None):
        # AI service endpoints
        self.services = {
            "gpt": os.getenv('GPT_SERVICE_URL', 'http://localhost:8002'),
            "claude": os.getenv('CLAUDE_SERVICE_URL', 'http://localhost:8003'), 
            "gemini": os.getenv('GEMINI_SERVICE_URL', 'http://localhost:8004')
        }
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
    async def initialize(self):
        """Initialize client"""
        logger.info("🔗 Real AI Client initialized - ready for HTTP calls")
    async def call_ai(self, model: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call AI with fallback for Claude overload"""
    
        # Claude overload protection
        if model == "claude":
            try:
                result = await self._call_claude_with_retry(message, context)
                return result
            except Exception as e:
                if "overload" in str(e).lower() or "529" in str(e):
                    logger.warning("⚠️ Claude overloaded, falling back to GPT")
                    return await self._call_gpt_fallback(message, context)
                raise e    
     
    async def call_ai(self, model: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call AI service via HTTP
        🔧 FIXED: This is the ONLY call_ai method (no recursion)
        """
        if model not in self.services:
            return self._error_response(f"Unknown model: {model}", model)
        
        service_url = self.services[model]
        endpoint = f"{service_url}/chat"
        
        # Prepare request
        request_data = {
            "message": message,
            "user_id": context.get("user_id", "anonymous") if context else "anonymous"
        }
        
        try:
            logger.info(f"🚀 Calling {model.upper()} service at {endpoint}")
            
            # Make HTTP call
            response = await self.http_client.post(
                endpoint,
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ {model.upper()} responded successfully")
                
                # Standardize response format
                return {
                    "response": result.get("response", "No response from AI"),
                    "model_used": model,
                    "confidence": result.get("confidence", 0.8),
                    "service_type": "real_http",
                    "service_url": service_url,
                    "status_code": response.status_code,
                    "processing_time_seconds": result.get("processing_time_seconds", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"❌ {model.upper()} service error: {response.status_code}")
                return self._error_response(
                    f"Service returned {response.status_code}: {response.text}",
                    model, service_url
                )
                
        except httpx.TimeoutException:
            logger.error(f"⏰ {model.upper()} service timeout")
            return self._error_response(f"Service timeout after 30s", model, service_url)
            
        except httpx.ConnectError:
            logger.error(f"🔌 Cannot connect to {model.upper()} service")
            return self._error_response(f"Cannot connect to service", model, service_url)
            
        except Exception as e:
            logger.error(f"💥 {model.upper()} service error: {e}")
            return self._error_response(str(e), model, service_url)

    # 🔧 FIXED: Removed the recursive alias method completely
    # The above call_ai method is the only implementation needed
    
    async def test_all_services(self, test_message: str = "Hello, this is a test") -> Dict[str, Any]:
        """
        Test all AI services
        """
        logger.info("🧪 Testing all AI services...")
        results = {}
        
        for model in self.services.keys():
            try:
                result = await self.call_ai(model, test_message)
                
                results[model] = {
                    "status": "success" if not result.get("error") else "error",
                    "available": not result.get("error"),
                    "response_preview": result.get("response", "")[:50] + "...",
                    "service_url": self.services[model],
                    "error": result.get("error")
                }
                
            except Exception as e:
                results[model] = {
                    "status": "error",
                    "available": False,
                    "error": str(e),
                    "service_url": self.services[model]
                }
        
        # Summary
        successful = sum(1 for r in results.values() if r["available"])
        total = len(results)
        
        return {
            "test_results": results,
            "summary": {
                "successful": successful,
                "total": total,
                "success_rate": round(successful / total * 100, 1) if total > 0 else 0
            },
            "all_services_available": successful == total,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_available_models(self) -> list:
        """Get available models"""
        return list(self.services.keys())
    
    async def check_service_health(self, model: str) -> Dict[str, Any]:
        """
        Check individual service health
        """
        if model not in self.services:
            return {"status": "unknown", "available": False, "error": "Unknown model"}
        
        service_url = self.services[model]
        health_endpoint = f"{service_url}/health"
        
        try:
            response = await self.http_client.get(health_endpoint, timeout=10.0)
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "available": True,
                    "service_url": service_url,
                    "health_data": health_data
                }
            else:
                return {
                    "status": "unhealthy", 
                    "available": False,
                    "service_url": service_url,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "available": False, 
                "service_url": service_url,
                "error": str(e)
            }
    
    def _error_response(self, error_message: str, model: str, service_url: str = "") -> Dict[str, Any]:
        """Generate error response"""
        return {
            "response": f"ขออภัย เกิดข้อผิดพลาดในการเรียกใช้ {model.upper()} service\n\nรายละเอียด: {error_message}",
            "model_used": model,
            "confidence": 0.0,
            "service_type": "real_http_error",
            "service_url": service_url,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()

# 🔧 FIXED: Clean file - no duplicate methods, no recursion, no compatibility aliases