"""
Claude Client Implementation with Enhanced Circuit Breaker
File: backend/services/03-external-ai/claude-client/app/claude_client.py
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import anthropic
from datetime import datetime
import time

# Import circuit breaker
import sys
sys.path.append('/app/backend/services/shared')
from common.enhanced_circuit_breaker import execute_with_circuit_breaker, circuit_breaker

logger = logging.getLogger(__name__)

class ClaudeClient:
    """
    Claude Client with circuit breaker protection and optimization
    """
    
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=20.0  # Default timeout
        )
        self.service_name = "claude"
        self.default_model = "claude-3-5-sonnet-20241022"
        self.setup_model_configurations()
        
    def setup_model_configurations(self):
        """Setup model-specific configurations"""
        self.model_configs = {
            "claude-3-5-sonnet-20241022": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "timeout": 20.0
            },
            "claude-3-opus-20240229": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "timeout": 30.0
            },
            "claude-3-haiku-20240307": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
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
                "max_tokens": 1024,
                "top_p": 0.95
            })
        elif task_type == "deep_analysis":
            optimized_config.update({
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.9
            })
        elif task_type == "creative":
            optimized_config.update({
                "temperature": 0.9,
                "max_tokens": 3072,
                "top_p": 0.95
            })
        elif task_type == "code_review":
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
    
    def format_messages_for_claude(self, messages: List[Dict[str, str]]) -> tuple:
        """Format messages for Claude API"""
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_message, conversation_messages
    
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
        timeout = final_config.pop('timeout', 20.0)
        
        async def claude_operation():
            start_time = time.time()
            
            try:
                # Format messages for Claude
                system_message, conversation_messages = self.format_messages_for_claude(messages)
                
                # Prepare API parameters
                api_params = {
                    "model": model,
                    "messages": conversation_messages,
                    **final_config
                }
                
                if system_message:
                    api_params["system"] = system_message
                
                response = await self.client.messages.create(**api_params)
                
                processing_time = time.time() - start_time
                
                # Format response
                result = {
                    "success": True,
                    "model": model,
                    "task_type": task_type,
                    "response": {
                        "content": response.content[0].text,
                        "role": "assistant",
                        "finish_reason": response.stop_reason
                    },
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    "metadata": {
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat(),
                        "model_config": optimized_config,
                        "service": "claude"
                    }
                }
                
                logger.info(f"Claude completion successful: {processing_time:.2f}s, {result['usage']['total_tokens']} tokens")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Claude API error after {processing_time:.2f}s: {e}")
                raise
        
        # Execute with circuit breaker
        try:
            return await execute_with_circuit_breaker(self.service_name, claude_operation)
        except Exception as e:
            # Return error response
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "task_type": task_type,
                "service": "claude",
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
    
    async def create_reasoning_enhanced(
        self,
        prompt: str,
        reasoning_type: str = "analytical",
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create completion with enhanced reasoning capabilities
        """
        reasoning_prompts = {
            "analytical": "Think through this step-by-step, analyzing each component carefully:",
            "creative": "Approach this creatively, considering multiple perspectives and innovative solutions:",
            "systematic": "Use systematic reasoning to break down this problem methodically:",
            "critical": "Apply critical thinking to evaluate this thoroughly:"
        }
        
        system_prompt = reasoning_prompts.get(reasoning_type, reasoning_prompts["analytical"])
        
        return await self.create_completion_with_system(
            system_prompt,
            prompt,
            model,
            task_type="deep_analysis",
            **kwargs
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for Claude service
        """
        try:
            # Simple health check with minimal token usage
            health_messages = [{"role": "user", "content": "Hello"}]
            
            start_time = time.time()
            response = await self.create_chat_completion(
                messages=health_messages,
                model="claude-3-haiku-20240307",  # Use faster model for health check
                task_type="conversation",
                max_tokens=10
            )
            response_time = time.time() - start_time
            
            if response["success"]:
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "service": "claude",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": response.get("error", "Unknown error"),
                    "response_time": response_time,
                    "service": "claude",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "service": "claude",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service": "claude",
            "provider": "Anthropic",
            "available_models": self.get_available_models(),
            "default_model": self.default_model,
            "circuit_breaker_status": circuit_breaker.get_service_health("claude"),
            "capabilities": [
                "chat_completion",
                "text_completion",
                "conversation",
                "deep_analysis",
                "reasoning_enhancement",
                "creative_writing",
                "code_review"
            ]
        }

# Global client instance
claude_client = ClaudeClient()

# Convenience functions for FastAPI endpoints
async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Chat completion endpoint"""
    return await claude_client.create_chat_completion(messages, model, task_type, **kwargs)

async def completion(
    prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion endpoint"""
    return await claude_client.create_completion(prompt, model, task_type, **kwargs)

async def completion_with_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    task_type: str = "conversation",
    **kwargs
) -> Dict[str, Any]:
    """Completion with system prompt endpoint"""
    return await claude_client.create_completion_with_system(system_prompt, user_prompt, model, task_type, **kwargs)

async def reasoning_enhanced(
    prompt: str,
    reasoning_type: str = "analytical",
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Enhanced reasoning endpoint"""
    return await claude_client.create_reasoning_enhanced(prompt, reasoning_type, model, **kwargs)

async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return await claude_client.health_check()

def get_service_info() -> Dict[str, Any]:
    """Service info endpoint"""
    return claude_client.get_service_info()

# Example usage and testing
async def example_usage():
    """Example usage of Claude client"""
    
    # Simple completion
    result1 = await completion("What is the capital of France?", task_type="conversation")
    print(f"Simple completion: {result1}")
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze the implications of quantum computing on cybersecurity."}
    ]
    result2 = await chat_completion(messages, task_type="deep_analysis")
    print(f"Chat completion: {result2}")
    
    # Reasoning enhanced
    result3 = await reasoning_enhanced(
        "How can we solve climate change?",
        reasoning_type="systematic"
    )
    print(f"Reasoning enhanced: {result3}")
    
    # Health check
    health = await health_check()
    print(f"Health check: {health}")
    
    # Service info
    info = get_service_info()
    print(f"Service info: {info}")

if __name__ == "__main__":
    # Test the Claude client
    asyncio.run(example_usage())