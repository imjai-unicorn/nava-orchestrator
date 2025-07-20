"""
GPT Client Implementation with Enhanced Circuit Breaker
File: backend/services/03-external-ai/gpt-client/app/gpt_client.py
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from datetime import datetime
import time

# Import circuit breaker
import sys
sys.path.append('/app/backend/services/shared')
from common.enhanced_circuit_breaker import execute_with_circuit_breaker, circuit_breaker

logger = logging.getLogger(__name__)

class GPTClient:
    """
    GPT Client with circuit breaker protection and optimization
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=15.0  # Default timeout
        )
        self.service_name = "gpt"
        self.default_model = "gpt-4o-mini"
        self.setup_model_configurations()
        
    def setup_model_configurations(self):
        """Setup model-specific configurations"""
        self.model_configs = {
            "gpt-4o-mini": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": 15.0
            },
            "gpt-4o": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": 30.0
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": 10.0
            }
        }
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        return self.model_configs.get(model, self.model_configs[self.default_model])
    
    def optimize_for_task(self, task_type: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model parameters for specific task types"""
        optimized_config = model_config.copy()
        
        if task_type == "conversation":
            optimized_config.update({
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.95
            })
        elif task_type == "creative":
            optimized_config.update({
                "temperature": 0.9,
                "max_tokens": 3072,
                "top_p": 0.95
            })
        elif task_type == "analytical":
            optimized_config.update({
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.9
            })
        elif task_type == "code":
            optimized_config.update({
                "temperature": 0.2,
                "max_tokens": 4096,
                "top_p": 0.95
            })
        elif task_type == "business":
            optimized_config.update({
                "temperature": 0.4,
                "max_tokens": 2048,
                "top_p": 0.9
            })
        
        return optimized_config
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        task_type: str = "conversation",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create chat completion with circuit breaker protection
        """
        model = model or self.default_model
        model_config = self.get_model_config(model)
        optimized_config = self.optimize_for_task(task_type, model_config)
        
        # Merge kwargs with optimized config
        final_config = {**optimized_config, **kwargs}
        
        # Remove timeout from API call parameters
        timeout = final_config.pop('timeout', 15.0)
        
        async def gpt_operation():
            start_time = time.time()
            
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **final_config
                )
                
                processing_time = time.time() - start_time
                
                # Format response
                result = {
                    "success": True,
                    "model": model,
                    "task_type": task_type,
                    "response": {
                        "content": response.choices[0].message.content,
                        "role": response.choices[0].message.role,
                        "finish_reason": response.choices[0].finish_reason
                    },
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "metadata": {
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat(),
                        "model_config": optimized_config,
                        "service": "gpt"
                    }
                }
                
                logger.info(f"GPT completion successful: {processing_time:.2f}s, {result['usage']['total_tokens']} tokens")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"GPT API error after {processing_time:.2f}s: {e}")
                raise
        
        # Execute with circuit breaker
        try:
            return await execute_with_circuit_breaker(self.service_name, gpt_operation)
        except Exception as e:
            # Return error response
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "task_type": task_type,
                "service": "gpt",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__
                }
            }
    
    async def create_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        task_type: str = "conversation",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create completion from prompt (converted to chat format)
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.create_chat_completion(messages, model, task_type, **kwargs)
    
    async def create_completion_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        task_type: str = "conversation",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create completion with system and user prompts
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return await self.create_chat_completion(messages, model, task_type, **kwargs)
    
    async def create_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        model: Optional[str] = None,
        task_type: str = "conversation",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Continue a conversation with history
        """
        return await self.create_chat_completion(conversation_history, model, task_type, **kwargs)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for GPT service
        """
        try:
            # Simple health check with minimal token usage
            health_messages = [{"role": "user", "content": "Hello"}]
            
            start_time = time.time()
            response = await self.create_chat_completion(
                messages=health_messages,
                model="gpt-3.5-turbo",  # Use cheaper model for health check
                task_type="conversation",
                max_tokens=10
            )
            response_time = time.time() - start_time
            
            if response["success"]:
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "service": "gpt",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": response.get("error", "Unknown error"),
                    "response_time": response_time,
                    "service": "gpt",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "service": "gpt",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service": "gpt",
            "provider": "OpenAI",
            "available_models": self.get_available_models(),
            "default_model": self.default_model,
            "circuit_breaker_status": circuit_breaker.get_service_health("gpt"),
            "capabilities": [
                "chat_completion",
                "text_completion",
                "conversation",
                "code_generation",
                "creative_writing",
                "analysis"
            ]
        }

# Global client instance
gpt_client = GPTClient()

# Convenience functions for FastAPI endpoints
async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Chat completion endpoint"""
    return await gpt_client.create_chat_completion(messages, model, task_type, **kwargs)

async def completion(
    prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion endpoint"""
    return await gpt_client.create_completion(prompt, model, task_type, **kwargs)

async def completion_with_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion with system prompt endpoint"""
    return await gpt_client.create_completion_with_system(system_prompt, user_prompt, model, task_type, **kwargs)

async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return await gpt_client.health_check()

def get_service_info() -> Dict[str, Any]:
    """Service info endpoint"""
    return gpt_client.get_service_info()

# Example usage and testing
async def example_usage():
    """Example usage of GPT client"""
    
    # Simple completion
    result1 = await completion("What is the capital of France?", task_type="conversation")
    print(f"Simple completion: {result1}")
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    result2 = await chat_completion(messages, task_type="analytical")
    print(f"Chat completion: {result2}")
    
    # Health check
    health = await health_check()
    print(f"Health check: {health}")
    
    # Service info
    info = get_service_info()
    print(f"Service info: {info}")

if __name__ == "__main__":
    # Test the GPT client
    asyncio.run(example_usage())
