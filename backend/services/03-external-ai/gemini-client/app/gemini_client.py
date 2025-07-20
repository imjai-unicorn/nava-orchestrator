"""
Gemini Client Implementation with Enhanced Circuit Breaker
File: backend/services/03-external-ai/gemini-client/app/gemini_client.py
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from datetime import datetime
import time

# Import circuit breaker
import sys
sys.path.append('/app/backend/services/shared')
from common.enhanced_circuit_breaker import execute_with_circuit_breaker, circuit_breaker

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Gemini Client with circuit breaker protection and optimization
    """
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.service_name = "gemini"
        self.default_model = "gemini-1.5-flash"
        self.setup_model_configurations()
        
    def setup_model_configurations(self):
        """Setup model-specific configurations"""
        self.model_configs = {
            "gemini-1.5-flash": {
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "timeout": 18.0
            },
            "gemini-1.5-pro": {
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "timeout": 25.0
            },
            "gemini-1.0-pro": {
                "max_output_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "timeout": 15.0
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
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40
            })
        elif task_type == "research":
            optimized_config.update({
                "temperature": 0.3,
                "max_output_tokens": 4096,
                "top_p": 0.9,
                "top_k": 20
            })
        elif task_type == "creative":
            optimized_config.update({
                "temperature": 0.9,
                "max_output_tokens": 3072,
                "top_p": 0.95,
                "top_k": 60
            })
        elif task_type == "document_processing":
            optimized_config.update({
                "temperature": 0.2,
                "max_output_tokens": 4096,
                "top_p": 0.9,
                "top_k": 20
            })
        elif task_type == "strategic_planning":
            optimized_config.update({
                "temperature": 0.4,
                "max_output_tokens": 2048,
                "top_p": 0.9,
                "top_k": 30
            })
        
        return optimized_config
    
    def format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Gemini API"""
        gemini_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have system role, prepend to first user message
                if not gemini_messages:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [msg["content"]]
                    })
                else:
                    # Insert system message into first user message
                    if gemini_messages[0]["role"] == "user":
                        gemini_messages[0]["parts"][0] = f"{msg['content']}\n\n{gemini_messages[0]['parts'][0]}"
            elif msg["role"] == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [msg["content"]]
                })
            elif msg["role"] == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })
        
        return gemini_messages
    
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
        timeout = final_config.pop('timeout', 18.0)
        
        async def gemini_operation():
            start_time = time.time()
            
            try:
                # Create generation config
                generation_config = genai.GenerationConfig(
                    max_output_tokens=final_config.get('max_output_tokens', 8192),
                    temperature=final_config.get('temperature', 0.7),
                    top_p=final_config.get('top_p', 0.95),
                    top_k=final_config.get('top_k', 40)
                )
                
                # Initialize model
                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )
                
                # Format messages for Gemini
                gemini_messages = self.format_messages_for_gemini(messages)
                
                # Start chat session
                if len(gemini_messages) == 1:
                    # Single message
                    response = await asyncio.wait_for(
                        asyncio.to_thread(gemini_model.generate_content, gemini_messages[0]["parts"][0]),
                        timeout=timeout
                    )
                else:
                    # Multi-turn conversation
                    chat = gemini_model.start_chat(history=gemini_messages[:-1])
                    response = await asyncio.wait_for(
                        asyncio.to_thread(chat.send_message, gemini_messages[-1]["parts"][0]),
                        timeout=timeout
                    )
                
                processing_time = time.time() - start_time
                
                # Extract response content
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # Format response
                result = {
                    "success": True,
                    "model": model,
                    "task_type": task_type,
                    "response": {
                        "content": response_text,
                        "role": "assistant",
                        "finish_reason": "stop"
                    },
                    "usage": {
                        "prompt_tokens": getattr(response, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response, 'candidates_token_count', 0) if hasattr(response, 'candidates_token_count') else len(response_text.split()),
                        "total_tokens": getattr(response, 'total_token_count', 0) if hasattr(response, 'total_token_count') else len(response_text.split())
                    },
                    "metadata": {
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat(),
                        "model_config": optimized_config,
                        "service": "gemini"
                    }
                }
                
                logger.info(f"Gemini completion successful: {processing_time:.2f}s, {result['usage']['total_tokens']} tokens")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Gemini API error after {processing_time:.2f}s: {e}")
                raise
        
        # Execute with circuit breaker
        try:
            return await execute_with_circuit_breaker(self.service_name, gemini_operation)
        except Exception as e:
            # Return error response
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "task_type": task_type,
                "service": "gemini",
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
    
    async def create_multimodal_completion(
        self,
        text_prompt: str,
        image_data: Optional[bytes] = None,
        model: Optional[str] = None,
        task_type: str = "multimodal",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create multimodal completion with text and image
        """
        model = model or self.default_model
        model_config = self.get_model_config(model)
        optimized_config = self.optimize_for_task(task_type, model_config)
        
        # Merge kwargs with optimized config
        final_config = {**optimized_config, **kwargs}
        timeout = final_config.pop('timeout', 18.0)
        
        async def multimodal_operation():
            start_time = time.time()
            
            try:
                # Create generation config
                generation_config = genai.GenerationConfig(
                    max_output_tokens=final_config.get('max_output_tokens', 8192),
                    temperature=final_config.get('temperature', 0.7),
                    top_p=final_config.get('top_p', 0.95),
                    top_k=final_config.get('top_k', 40)
                )
                
                # Initialize model
                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )
                
                # Prepare content
                content = [text_prompt]
                if image_data:
                    # Add image to content
                    content.append({"mime_type": "image/jpeg", "data": image_data})
                
                # Generate response
                response = await asyncio.wait_for(
                    asyncio.to_thread(gemini_model.generate_content, content),
                    timeout=timeout
                )
                
                processing_time = time.time() - start_time
                
                # Extract response content
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # Format response
                result = {
                    "success": True,
                    "model": model,
                    "task_type": task_type,
                    "response": {
                        "content": response_text,
                        "role": "assistant",
                        "finish_reason": "stop"
                    },
                    "usage": {
                        "prompt_tokens": getattr(response, 'prompt_token_count', 0),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(response_text.split())
                    },
                    "metadata": {
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat(),
                        "model_config": optimized_config,
                        "service": "gemini",
                        "multimodal": True
                    }
                }
                
                logger.info(f"Gemini multimodal completion successful: {processing_time:.2f}s")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Gemini multimodal API error after {processing_time:.2f}s: {e}")
                raise
        
        # Execute with circuit breaker
        try:
            return await execute_with_circuit_breaker(self.service_name, multimodal_operation)
        except Exception as e:
            # Return error response
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "task_type": task_type,
                "service": "gemini",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__,
                    "multimodal": True
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for Gemini service
        """
        try:
            # Simple health check with minimal token usage
            start_time = time.time()
            response = await self.create_completion(
                "Hello",
                model="gemini-1.0-pro",  # Use basic model for health check
                task_type="conversation",
                max_output_tokens=10
            )
            response_time = time.time() - start_time
            
            if response["success"]:
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "service": "gemini",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": response.get("error", "Unknown error"),
                    "response_time": response_time,
                    "service": "gemini",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "service": "gemini",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service": "gemini",
            "provider": "Google",
            "available_models": self.get_available_models(),
            "default_model": self.default_model,
            "circuit_breaker_status": circuit_breaker.get_service_health("gemini"),
            "capabilities": [
                "chat_completion",
                "text_completion",
                "conversation",
                "multimodal_processing",
                "document_processing",
                "research",
                "strategic_planning"
            ]
        }

# Global client instance
gemini_client = GeminiClient()

# Convenience functions for FastAPI endpoints
async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Chat completion endpoint"""
    return await gemini_client.create_chat_completion(messages, model, task_type, **kwargs)

async def completion(
    prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion endpoint"""
    return await gemini_client.create_completion(prompt, model, task_type, **kwargs)

async def completion_with_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion with system prompt endpoint"""
    return await gemini_client.create_completion_with_system(system_prompt, user_prompt, model, task_type, **kwargs)

async def multimodal_completion(
    text_prompt: str,
    image_data: Optional[bytes] = None,
    model: Optional[str] = None,
    task_type: str = "multimodal",
    **kwargs
) -> Dict[str, Any]:
    """Multimodal completion endpoint"""
    return await gemini_client.create_multimodal_completion(text_prompt, image_data, model, task_type, **kwargs)

async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return await gemini_client.health_check()

def get_service_info() -> Dict[str, Any]:
    """Service info endpoint"""
    return gemini_client.get_service_info()

# Example usage and testing
async def example_usage():
    """Example usage of Gemini client"""
    
    # Simple completion
    result1 = await completion("What is the capital of France?", task_type="conversation")
    print(f"Simple completion: {result1}")
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": "Research the latest developments in renewable energy technology."}
    ]
    result2 = await chat_completion(messages, task_type="research")
    print(f"Chat completion: {result2}")
    
    # Multimodal completion (example without actual image)
    result3 = await multimodal_completion(
        "Describe this image",
        image_data=None,  # Would contain actual image bytes
        task_type="multimodal"
    )
    print(f"Multimodal completion: {result3}")
    
    # Health check
    health = await health_check()
    print(f"Health check: {health}")
    
    # Service info
    info = get_service_info()
    print(f"Service info: {info}")

if __name__ == "__main__":
    # Test the Gemini client
    asyncio.run(example_usage())