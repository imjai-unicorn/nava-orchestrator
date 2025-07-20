#!/usr/bin/env python3
"""
GPT Client Unit Tests
File: backend/services/03-external-ai/gpt-client/tests/test_gpt_client.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add app path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from gpt_client import GPTClient, chat_completion, completion, health_check, get_service_info

class TestGPTClient:
    """Test GPT Client functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = GPTClient()
    
    def test_client_initialization(self):
        """Test client initialization"""
        assert self.client.service_name == "gpt"
        assert self.client.default_model == "gpt-4o-mini"
        assert len(self.client.model_configs) >= 3
        
        # Check default model config
        config = self.client.get_model_config("gpt-4o-mini")
        assert config["max_tokens"] == 4096
        assert config["temperature"] == 0.7
        assert config["timeout"] == 15.0
    
    def test_model_configurations(self):
        """Test model configurations"""
        models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        
        for model in models:
            config = self.client.get_model_config(model)
            assert "max_tokens" in config
            assert "temperature" in config
            assert "timeout" in config
            assert isinstance(config["max_tokens"], int)
            assert isinstance(config["temperature"], float)
            assert isinstance(config["timeout"], float)
    
    def test_task_optimization(self):
        """Test task-specific optimization"""
        base_config = self.client.get_model_config("gpt-4o-mini")
        
        # Test conversation optimization
        conv_config = self.client.optimize_for_task("conversation", base_config)
        assert conv_config["temperature"] == 0.7
        assert conv_config["max_tokens"] == 1024
        
        # Test creative optimization
        creative_config = self.client.optimize_for_task("creative", base_config)
        assert creative_config["temperature"] == 0.9
        assert creative_config["max_tokens"] == 3072
        
        # Test analytical optimization
        analytical_config = self.client.optimize_for_task("analytical", base_config)
        assert analytical_config["temperature"] == 0.3
        assert analytical_config["max_tokens"] == 4096
        
        # Test code optimization
        code_config = self.client.optimize_for_task("code", base_config)
        assert code_config["temperature"] == 0.2
        assert code_config["max_tokens"] == 4096
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_create_chat_completion_success(self, mock_openai):
        """Test successful chat completion"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test chat completion
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["model"] == "gpt-4o-mini"
        assert result["response"]["content"] == "Test response"
        assert result["response"]["role"] == "assistant"
        assert result["usage"]["total_tokens"] == 15
        assert "metadata" in result
        assert "processing_time" in result["metadata"]
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_create_chat_completion_failure(self, mock_openai):
        """Test chat completion failure"""
        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test chat completion failure
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "error" in result
        assert result["error"] == "API Error"
        assert result["service"] == "gpt"
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_create_completion(self, mock_openai):
        """Test simple completion"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test completion
        result = await client.create_completion("Hello")
        
        assert result["success"] == True
        assert result["response"]["content"] == "Test response"
        assert result["usage"]["total_tokens"] == 8
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_create_completion_with_system(self, mock_openai):
        """Test completion with system prompt"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "System response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 25
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test completion with system prompt
        result = await client.create_completion_with_system(
            "You are a helpful assistant",
            "Hello"
        )
        
        assert result["success"] == True
        assert result["response"]["content"] == "System response"
        
        # Verify system message was included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_health_check(self, mock_openai):
        """Test health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test health check
        result = await client.health_check()
        
        assert result["status"] == "healthy"
        assert result["service"] == "gpt"
        assert "response_time" in result
        assert "timestamp" in result
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_openai):
        """Test health check failure"""
        # Mock failed response
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Connection failed"))
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test health check failure
        result = await client.health_check()
        
        assert result["status"] == "error"
        assert result["service"] == "gpt"
        assert "error" in result
        assert "Connection failed" in result["error"]
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = self.client.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) >= 3
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models
    
    def test_get_service_info(self):
        """Test getting service information"""
        info = self.client.get_service_info()
        
        assert info["service"] == "gpt"
        assert info["provider"] == "OpenAI"
        assert "available_models" in info
        assert info["default_model"] == "gpt-4o-mini"
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)
    
    @pytest.mark.asyncio
    async def test_different_task_types(self):
        """Test different task types"""
        with patch('gpt_client.AsyncOpenAI') as mock_openai:
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.role = "assistant"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client
            
            # Re-initialize client with mock
            client = GPTClient()
            client.client = mock_client
            
            # Test different task types
            task_types = ["conversation", "creative", "analytical", "code", "business"]
            messages = [{"role": "user", "content": "Test"}]
            
            for task_type in task_types:
                result = await client.create_chat_completion(messages, task_type=task_type)
                assert result["success"] == True
                assert result["task_type"] == task_type
                
                # Check that task-specific optimization was applied
                call_args = mock_client.chat.completions.create.call_args
                if task_type == "creative":
                    assert call_args[1]["temperature"] == 0.9
                elif task_type == "analytical":
                    assert call_args[1]["temperature"] == 0.3
                elif task_type == "code":
                    assert call_args[1]["temperature"] == 0.2

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('gpt_client.gpt_client')
    @pytest.mark.asyncio
    async def test_chat_completion_function(self, mock_client):
        """Test chat completion convenience function"""
        # Mock client method
        mock_client.create_chat_completion = AsyncMock(return_value={
            "success": True,
            "response": {"content": "Test response"}
        })
        
        # Test function
        messages = [{"role": "user", "content": "Hello"}]
        result = await chat_completion(messages)
        
        assert result["success"] == True
        assert result["response"]["content"] == "Test response"
        
        # Verify client method was called
        mock_client.create_chat_completion.assert_called_once_with(
            messages, None, "conversation"
        )
    
    @patch('gpt_client.gpt_client')
    @pytest.mark.asyncio
    async def test_completion_function(self, mock_client):
        """Test completion convenience function"""
        # Mock client method
        mock_client.create_completion = AsyncMock(return_value={
            "success": True,
            "response": {"content": "Test response"}
        })
        
        # Test function
        result = await completion("Hello")
        
        assert result["success"] == True
        assert result["response"]["content"] == "Test response"
        
        # Verify client method was called
        mock_client.create_completion.assert_called_once_with(
            "Hello", None, "conversation"
        )
    
    @patch('gpt_client.gpt_client')
    @pytest.mark.asyncio
    async def test_health_check_function(self, mock_client):
        """Test health check convenience function"""
        # Mock client method
        mock_client.health_check = AsyncMock(return_value={
            "status": "healthy",
            "service": "gpt"
        })
        
        # Test function
        result = await health_check()
        
        assert result["status"] == "healthy"
        assert result["service"] == "gpt"
        
        # Verify client method was called
        mock_client.health_check.assert_called_once()
    
    @patch('gpt_client.gpt_client')
    def test_get_service_info_function(self, mock_client):
        """Test get service info convenience function"""
        # Mock client method
        mock_client.get_service_info = Mock(return_value={
            "service": "gpt",
            "provider": "OpenAI"
        })
        
        # Test function
        result = get_service_info()
        
        assert result["service"] == "gpt"
        assert result["provider"] == "OpenAI"
        
        # Verify client method was called
        mock_client.get_service_info.assert_called_once()

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_openai):
        """Test timeout handling"""
        # Mock timeout exception
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError("Request timeout"))
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test timeout
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "timeout" in result["error"].lower()
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_openai):
        """Test API error handling"""
        # Mock API exception
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test API error
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "Rate limit exceeded" in result["error"]
        assert result["metadata"]["error_type"] == "Exception"
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_invalid_model_handling(self, mock_openai):
        """Test invalid model handling"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test with invalid model (should fall back to default)
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages, model="invalid-model")
        
        assert result["success"] == True
        # Should fall back to default model config
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "invalid-model"  # Model name passed through
        # But config should be from default model
        assert call_args[1]["max_tokens"] == 4096  # Default model's max_tokens

class TestPerformanceMetrics:
    """Test performance metrics tracking"""
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, mock_openai):
        """Test processing time tracking"""
        # Mock OpenAI response with delay
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.role = "assistant"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            return mock_response
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=delayed_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test processing time tracking
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert "processing_time" in result["metadata"]
        assert result["metadata"]["processing_time"] > 0
        assert result["metadata"]["processing_time"] < 1  # Should be less than 1 second
    
    @patch('gpt_client.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_usage_tracking(self, mock_openai):
        """Test token usage tracking"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 40
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Re-initialize client with mock
        client = GPTClient()
        client.client = mock_client
        
        # Test usage tracking
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["usage"]["prompt_tokens"] == 25
        assert result["usage"]["completion_tokens"] == 15
        assert result["usage"]["total_tokens"] == 40

if __name__ == "__main__":
    pytest.main([__file__, "-v"])