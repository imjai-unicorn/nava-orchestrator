#!/usr/bin/env python3
"""
Claude Client Unit Tests
File: backend/services/03-external-ai/claude-client/tests/test_claude_client.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add app path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from claude_client import ClaudeClient, chat_completion, completion, reasoning_enhanced, health_check, get_service_info

class TestClaudeClient:
    """Test Claude Client functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = ClaudeClient()
    
    def test_client_initialization(self):
        """Test client initialization"""
        assert self.client.service_name == "claude"
        assert self.client.default_model == "claude-3-5-sonnet-20241022"
        assert len(self.client.model_configs) >= 3
        
        # Check default model config
        config = self.client.get_model_config("claude-3-5-sonnet-20241022")
        assert config["max_tokens"] == 4096
        assert config["temperature"] == 0.7
        assert config["timeout"] == 20.0
    
    def test_model_configurations(self):
        """Test model configurations"""
        models = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        
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
        base_config = self.client.get_model_config("claude-3-5-sonnet-20241022")
        
        # Test conversation optimization
        conv_config = self.client.optimize_for_task("conversation", base_config)
        assert conv_config["temperature"] == 0.7
        assert conv_config["max_tokens"] == 1024
        
        # Test deep analysis optimization
        analysis_config = self.client.optimize_for_task("deep_analysis", base_config)
        assert analysis_config["temperature"] == 0.3
        assert analysis_config["max_tokens"] == 4096
        
        # Test creative optimization
        creative_config = self.client.optimize_for_task("creative", base_config)
        assert creative_config["temperature"] == 0.9
        assert creative_config["max_tokens"] == 3072
        
        # Test code review optimization
        code_config = self.client.optimize_for_task("code_review", base_config)
        assert code_config["temperature"] == 0.2
        assert code_config["max_tokens"] == 4096
    
    def test_message_formatting(self):
        """Test message formatting for Claude API"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        system_message, conversation_messages = self.client.format_messages_for_claude(messages)
        
        assert system_message == "You are a helpful assistant"
        assert len(conversation_messages) == 3
        assert conversation_messages[0]["role"] == "user"
        assert conversation_messages[1]["role"] == "assistant"
        assert conversation_messages[2]["role"] == "user"
    
    @patch('claude_client.anthropic.AsyncAnthropic')
    @pytest.mark.asyncio
    async def test_create_chat_completion_success(self, mock_anthropic):
        """Test successful chat completion"""
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test Claude response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 10
        
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        # Re-initialize client with mock
        client = ClaudeClient()
        client.client = mock_client
        
        # Test chat completion
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["response"]["content"] == "Test Claude response"
        assert result["response"]["role"] == "assistant"
        assert result["usage"]["total_tokens"] == 25
        assert "metadata" in result
        assert "processing_time" in result["metadata"]
    
    @patch('claude_client.anthropic.AsyncAnthropic')
    @pytest.mark.asyncio
    async def test_create_chat_completion_failure(self, mock_anthropic):
        """Test chat completion failure"""
        # Mock Anthropic client to raise exception
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        mock_anthropic.return_value = mock_client
        
        # Re-initialize client with mock
        client = ClaudeClient()
        client.client = mock_client
        
        # Test chat completion failure
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "error" in result
        assert result["error"] == "API Error"
        assert result["service"] == "claude"
    
    @patch('claude_client.anthropic.AsyncAnthropic')
    @pytest.mark.asyncio
    async def test_create_reasoning_enhanced(self, mock_anthropic):
        """Test reasoning enhanced completion"""
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Enhanced reasoning response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 15
        
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        # Re-initialize client with mock
        client = ClaudeClient()
        client.client = mock_client
        
        # Test reasoning enhanced
        result = await client.create_reasoning_enhanced(
            "Analyze this complex problem",
            reasoning_type="systematic"
        )
        
        assert result["success"] == True
        assert result["response"]["content"] == "Enhanced reasoning response"
        assert result["task_type"] == "deep_analysis"
        
        # Verify system prompt was used
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args[1]
        assert "systematic" in call_args[1]["system"].lower()
    
    @patch('claude_client.anthropic.AsyncAnthropic')
    @pytest.mark.asyncio
    async def test_health_check(self, mock_anthropic):
        """Test health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Hello"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        # Re-initialize client with mock
        client = ClaudeClient()
        client.client = mock_client
        
        # Test health check
        result = await client.health_check()
        
        assert result["status"] == "healthy"
        assert result["service"] == "claude"
        assert "response_time" in result
        assert "timestamp" in result
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = self.client.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) >= 3
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-opus-20240229" in models
        assert "claude-3-haiku-20240307" in models
    
    def test_get_service_info(self):
        """Test getting service information"""
        info = self.client.get_service_info()
        
        assert info["service"] == "claude"
        assert info["provider"] == "Anthropic"
        assert "available_models" in info
        assert info["default_model"] == "claude-3-5-sonnet-20241022"
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)
        assert "deep_analysis" in info["capabilities"]
    
    @pytest.mark.asyncio
    async def test_different_reasoning_types(self):
        """Test different reasoning types"""
        with patch('claude_client.anthropic.AsyncAnthropic') as mock_anthropic:
            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Reasoning response"
            mock_response.stop_reason = "end_turn"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            
            mock_client = Mock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            
            # Re-initialize client with mock
            client = ClaudeClient()
            client.client = mock_client
            
            # Test different reasoning types
            reasoning_types = ["analytical", "creative", "systematic", "critical"]
            
            for reasoning_type in reasoning_types:
                result = await client.create_reasoning_enhanced(
                    "Test prompt",
                    reasoning_type=reasoning_type
                )
                assert result["success"] == True
                assert result["task_type"] == "deep_analysis"

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('claude_client.claude_client')
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
    
    @patch('claude_client.claude_client')
    @pytest.mark.asyncio
    async def test_reasoning_enhanced_function(self, mock_client):
        """Test reasoning enhanced convenience function"""
        # Mock client method
        mock_client.create_reasoning_enhanced = AsyncMock(return_value={
            "success": True,
            "response": {"content": "Enhanced response"}
        })
        
        # Test function
        result = await reasoning_enhanced("Test prompt", "analytical")
        
        assert result["success"] == True
        assert result["response"]["content"] == "Enhanced response"
        
        # Verify client method was called
        mock_client.create_reasoning_enhanced.assert_called_once_with(
            "Test prompt", "analytical", None
        )

class TestClaudeSpecificFeatures:
    """Test Claude-specific features"""
    
    def test_system_message_handling(self):
        """Test system message handling"""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        client = ClaudeClient()
        system_message, conversation_messages = client.format_messages_for_claude(messages)
        
        assert system_message == "Be helpful"
        assert len(conversation_messages) == 1
        assert conversation_messages[0]["role"] == "user"
    
    def test_no_system_message_handling(self):
        """Test handling when no system message"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        client = ClaudeClient()
        system_message, conversation_messages = client.format_messages_for_claude(messages)
        
        assert system_message == ""
        assert len(conversation_messages) == 2
    
    @patch('claude_client.anthropic.AsyncAnthropic')
    @pytest.mark.asyncio
    async def test_claude_specific_parameters(self, mock_anthropic):
        """Test Claude-specific API parameters"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        # Re-initialize client with mock
        client = ClaudeClient()
        client.client = mock_client
        
        # Test with system message
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        await client.create_chat_completion(messages)
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args[1]
        assert call_args[1]["system"] == "Be helpful"
        assert "messages" in call_args[1]
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
