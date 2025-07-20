#!/usr/bin/env python3
"""
Gemini Client Unit Tests
File: backend/services/03-external-ai/gemini-client/tests/test_gemini_client.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add app path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from gemini_client import GeminiClient, chat_completion, completion, multimodal_completion, health_check, get_service_info

class TestGeminiClient:
    """Test Gemini Client functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = GeminiClient()
    
    def test_client_initialization(self):
        """Test client initialization"""
        assert self.client.service_name == "gemini"
        assert self.client.default_model == "gemini-1.5-flash"
        assert len(self.client.model_configs) >= 3
        
        # Check default model config
        config = self.client.get_model_config("gemini-1.5-flash")
        assert config["max_output_tokens"] == 8192
        assert config["temperature"] == 0.7
        assert config["timeout"] == 18.0
    
    def test_model_configurations(self):
        """Test model configurations"""
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
        
        for model in models:
            config = self.client.get_model_config(model)
            assert "max_output_tokens" in config
            assert "temperature" in config
            assert "timeout" in config
            assert "top_p" in config
            assert "top_k" in config
            assert isinstance(config["max_output_tokens"], int)
            assert isinstance(config["temperature"], float)
            assert isinstance(config["timeout"], float)
    
    def test_task_optimization(self):
        """Test task-specific optimization"""
        base_config = self.client.get_model_config("gemini-1.5-flash")
        
        # Test conversation optimization
        conv_config = self.client.optimize_for_task("conversation", base_config)
        assert conv_config["temperature"] == 0.7
        assert conv_config["max_output_tokens"] == 1024
        assert conv_config["top_k"] == 40
        
        # Test research optimization
        research_config = self.client.optimize_for_task("research", base_config)
        assert research_config["temperature"] == 0.3
        assert research_config["max_output_tokens"] == 4096
        assert research_config["top_k"] == 20
        
        # Test creative optimization
        creative_config = self.client.optimize_for_task("creative", base_config)
        assert creative_config["temperature"] == 0.9
        assert creative_config["max_output_tokens"] == 3072
        assert creative_config["top_k"] == 60
        
        # Test document processing optimization
        doc_config = self.client.optimize_for_task("document_processing", base_config)
        assert doc_config["temperature"] == 0.2
        assert doc_config["max_output_tokens"] == 4096
        assert doc_config["top_k"] == 20
    
    def test_message_formatting(self):
        """Test message formatting for Gemini API"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        gemini_messages = self.client.format_messages_for_gemini(messages)
        
        assert len(gemini_messages) == 3
        # System message should be prepended to first user message
        assert "You are a helpful assistant" in gemini_messages[0]["parts"][0]
        assert "Hello" in gemini_messages[0]["parts"][0]
        assert gemini_messages[1]["role"] == "model"
        assert gemini_messages[2]["role"] == "user"
    
    def test_message_formatting_no_system(self):
        """Test message formatting without system message"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        gemini_messages = self.client.format_messages_for_gemini(messages)
        
        assert len(gemini_messages) == 2
        assert gemini_messages[0]["role"] == "user"
        assert gemini_messages[0]["parts"] == ["Hello"]
        assert gemini_messages[1]["role"] == "model"
        assert gemini_messages[1]["parts"] == ["Hi there!"]
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_create_chat_completion_success(self, mock_genai):
        """Test successful chat completion"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Test Gemini response"
        mock_response.prompt_token_count = 12
        mock_response.candidates_token_count = 8
        mock_response.total_token_count = 20
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test chat completion
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["model"] == "gemini-1.5-flash"
        assert result["response"]["content"] == "Test Gemini response"
        assert result["response"]["role"] == "assistant"
        assert result["usage"]["total_tokens"] == 20
        assert "metadata" in result
        assert "processing_time" in result["metadata"]
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_create_chat_completion_failure(self, mock_genai):
        """Test chat completion failure"""
        # Mock Gemini to raise exception
        mock_model = Mock()
        mock_model.generate_content = Mock(side_effect=Exception("API Error"))
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test chat completion failure
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "error" in result
        assert result["error"] == "API Error"
        assert result["service"] == "gemini"
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_create_multimodal_completion(self, mock_genai):
        """Test multimodal completion"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Multimodal response"
        mock_response.prompt_token_count = 15
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test multimodal completion
        result = await self.client.create_multimodal_completion(
            "Describe this image",
            image_data=b"fake_image_data"
        )
        
        assert result["success"] == True
        assert result["response"]["content"] == "Multimodal response"
        assert result["metadata"]["multimodal"] == True
        assert result["task_type"] == "multimodal"
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_create_multimodal_completion_text_only(self, mock_genai):
        """Test multimodal completion with text only"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Text only response"
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test multimodal completion without image
        result = await self.client.create_multimodal_completion(
            "Answer this question",
            image_data=None
        )
        
        assert result["success"] == True
        assert result["response"]["content"] == "Text only response"
        assert result["metadata"]["multimodal"] == True
        
        # Verify content was text only
        call_args = mock_model.generate_content.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0] == "Answer this question"
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_genai):
        """Test multi-turn conversation"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Conversation response"
        
        mock_chat = Mock()
        mock_chat.send_message = Mock(return_value=mock_response)
        
        mock_model = Mock()
        mock_model.start_chat = Mock(return_value=mock_chat)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test multi-turn conversation
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["response"]["content"] == "Conversation response"
        
        # Verify chat was started with history
        mock_model.start_chat.assert_called_once()
        mock_chat.send_message.assert_called_once()
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_health_check(self, mock_genai):
        """Test health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Hello"
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test health check
        result = await self.client.health_check()
        
        assert result["status"] == "healthy"
        assert result["service"] == "gemini"
        assert "response_time" in result
        assert "timestamp" in result
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = self.client.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) >= 3
        assert "gemini-1.5-flash" in models
        assert "gemini-1.5-pro" in models
        assert "gemini-1.0-pro" in models
    
    def test_get_service_info(self):
        """Test getting service information"""
        info = self.client.get_service_info()
        
        assert info["service"] == "gemini"
        assert info["provider"] == "Google"
        assert "available_models" in info
        assert info["default_model"] == "gemini-1.5-flash"
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)
        assert "multimodal_processing" in info["capabilities"]
    
    @pytest.mark.asyncio
    async def test_different_task_types(self):
        """Test different task types"""
        with patch('gemini_client.genai') as mock_genai:
            # Mock response
            mock_response = Mock()
            mock_response.text = "Task response"
            
            mock_model = Mock()
            mock_model.generate_content = Mock(return_value=mock_response)
            mock_genai.GenerativeModel = Mock(return_value=mock_model)
            mock_genai.GenerationConfig = Mock()
            
            # Test different task types
            task_types = ["conversation", "research", "creative", "document_processing", "strategic_planning"]
            messages = [{"role": "user", "content": "Test"}]
            
            for task_type in task_types:
                result = await self.client.create_chat_completion(messages, task_type=task_type)
                assert result["success"] == True
                assert result["task_type"] == task_type

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('gemini_client.gemini_client')
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
    
    @patch('gemini_client.gemini_client')
    @pytest.mark.asyncio
    async def test_multimodal_completion_function(self, mock_client):
        """Test multimodal completion convenience function"""
        # Mock client method
        mock_client.create_multimodal_completion = AsyncMock(return_value={
            "success": True,
            "response": {"content": "Multimodal response"}
        })
        
        # Test function
        result = await multimodal_completion("Describe this", b"image_data")
        
        assert result["success"] == True
        assert result["response"]["content"] == "Multimodal response"
        
        # Verify client method was called
        mock_client.create_multimodal_completion.assert_called_once_with(
            "Describe this", b"image_data", None, "multimodal"
        )

class TestGeminiSpecificFeatures:
    """Test Gemini-specific features"""
    
    def test_generation_config_creation(self):
        """Test generation config creation"""
        client = GeminiClient()
        config = client.get_model_config("gemini-1.5-flash")
        
        # Test that all required parameters are present
        assert "max_output_tokens" in config
        assert "temperature" in config
        assert "top_p" in config
        assert "top_k" in config
        
        # Test parameter ranges
        assert 0 <= config["temperature"] <= 1
        assert 0 <= config["top_p"] <= 1
        assert config["top_k"] > 0
        assert config["max_output_tokens"] > 0
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_genai):
        """Test timeout handling"""
        # Mock timeout scenario
        mock_model = Mock()
        mock_model.generate_content = Mock(side_effect=asyncio.TimeoutError("Request timeout"))
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test timeout
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "timeout" in result["error"].lower()
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_gemini_response_parsing(self, mock_genai):
        """Test Gemini response parsing"""
        # Mock response with different attributes
        mock_response = Mock()
        mock_response.text = "Parsed response"
        # Some responses might not have token counts
        delattr(mock_response, 'prompt_token_count') if hasattr(mock_response, 'prompt_token_count') else None
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test response parsing
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["response"]["content"] == "Parsed response"
        # Should handle missing token counts gracefully
        assert "usage" in result
    
    def test_role_mapping(self):
        """Test role mapping for Gemini"""
        messages = [
            {"role": "assistant", "content": "I'm an assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        client = GeminiClient()
        gemini_messages = client.format_messages_for_gemini(messages)
        
        # Assistant role should be mapped to "model"
        assert gemini_messages[0]["role"] == "model"
        assert gemini_messages[1]["role"] == "user"
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_image_processing(self, mock_genai):
        """Test image processing in multimodal completion"""
        # Mock response
        mock_response = Mock()
        mock_response.text = "Image analysis response"
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test with image data
        result = await self.client.create_multimodal_completion(
            "What's in this image?",
            image_data=b"fake_jpeg_data"
        )
        
        assert result["success"] == True
        assert result["response"]["content"] == "Image analysis response"
        
        # Verify content included both text and image
        call_args = mock_model.generate_content.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 2  # Text + image
        assert call_args[0] == "What's in this image?"
        assert isinstance(call_args[1], dict)
        assert call_args[1]["mime_type"] == "image/jpeg"

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_genai):
        """Test API error handling"""
        # Mock API exception
        mock_model = Mock()
        mock_model.generate_content = Mock(side_effect=Exception("API quota exceeded"))
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test API error
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == False
        assert "API quota exceeded" in result["error"]
        assert result["metadata"]["error_type"] == "Exception"
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_invalid_model_handling(self, mock_genai):
        """Test invalid model handling"""
        # Mock response
        mock_response = Mock()
        mock_response.text = "Response"
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test with invalid model (should fall back to default config)
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages, model="invalid-model")
        
        assert result["success"] == True
        # Should fall back to default model config
        assert result["model"] == "invalid-model"  # Model name passed through
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, mock_genai):
        """Test empty response handling"""
        # Mock empty response
        mock_response = Mock()
        mock_response.text = ""
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test empty response
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["response"]["content"] == ""

class TestPerformanceMetrics:
    """Test performance metrics tracking"""
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, mock_genai):
        """Test processing time tracking"""
        # Mock response with delay
        async def delayed_generate(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            mock_response = Mock()
            mock_response.text = "Delayed response"
            return mock_response
        
        mock_model = Mock()
        mock_model.generate_content = Mock(side_effect=delayed_generate)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test processing time tracking
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert "processing_time" in result["metadata"]
        assert result["metadata"]["processing_time"] > 0
        assert result["metadata"]["processing_time"] < 1  # Should be less than 1 second
    
    @patch('gemini_client.genai')
    @pytest.mark.asyncio
    async def test_token_usage_tracking(self, mock_genai):
        """Test token usage tracking"""
        # Mock response with token counts
        mock_response = Mock()
        mock_response.text = "Token test response"
        mock_response.prompt_token_count = 30
        mock_response.candidates_token_count = 20
        mock_response.total_token_count = 50
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        mock_genai.GenerativeModel = Mock(return_value=mock_model)
        mock_genai.GenerationConfig = Mock()
        
        # Test token usage tracking
        messages = [{"role": "user", "content": "Hello"}]
        result = await self.client.create_chat_completion(messages)
        
        assert result["success"] == True
        assert result["usage"]["prompt_tokens"] == 30
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 50

if __name__ == "__main__":
    pytest.main([__file__, "-v"])