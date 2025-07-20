# backend/services/01-core/nava-logic-controller/tests/test_enhanced_chat.py
"""
Test suite for enhanced chat models
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

# Import the models we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app', 'models'))

from chat import (
    MessageType, ConversationStatus, MessagePriority,
    ChatContext, Message, ChatRequest, ChatResponse, Conversation,
    ConversationSummary, ChatFeedback, ChatAnalytics,
    create_chat_context, create_message, calculate_conversation_metrics,
    validate_chat_request
)

class TestChatContext:
    """Test ChatContext model"""
    
    def test_chat_context_creation(self):
        """Test creating a ChatContext instance"""
        context = ChatContext(
            user_id="user_123",
            session_id="session_456",
            conversation_id="conv_789",
            user_role="developer",
            platform="web",
            organization_id="org_001"
        )
        
        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.conversation_id == "conv_789"
        assert context.user_role == "developer"
        assert context.platform == "web"
        assert context.organization_id == "org_001"
        assert isinstance(context.created_at, datetime)
    
    def test_chat_context_with_preferences(self):
        """Test chat context with user preferences"""
        preferences = {
            "language": "en",
            "response_style": "detailed",
            "technical_level": "expert"
        }
        
        context = ChatContext(
            user_id="user_123",
            session_id="session_456",
            user_preferences=preferences
        )
        
        assert context.user_preferences == preferences
        assert context.user_preferences["language"] == "en"

class TestMessage:
    """Test Message model"""
    
    def test_message_creation_user(self):
        """Test creating a user message"""
        message = Message(
            message_id="msg_001",
            conversation_id="conv_123",
            message_type=MessageType.USER,
            content="Hello, how can you help me?",
            sender_type="user",
            sender_id="user_456"
        )
        
        assert message.message_id == "msg_001"
        assert message.message_type == MessageType.USER
        assert message.content == "Hello, how can you help me?"
        assert message.sender_type == "user"
        assert message.priority == MessagePriority.NORMAL
        assert isinstance(message.created_at, datetime)
    
    def test_message_creation_assistant(self):
        """Test creating an assistant message"""
        message = Message(
            message_id="msg_002",
            conversation_id="conv_123",
            message_type=MessageType.ASSISTANT,
            content="I can help you with various tasks...",
            sender_type="ai",
            model_used="gpt-4",
            confidence_score=0.92,
            processing_time=1.5,
            tokens_used=150
        )
        
        assert message.message_type == MessageType.ASSISTANT
        assert message.model_used == "gpt-4"
        assert message.confidence_score == 0.92
        assert message.processing_time == 1.5
        assert message.tokens_used == 150
    
    def test_message_with_reasoning_trace(self):
        """Test message with AI reasoning trace"""
        reasoning = {
            "decision_factors": ["user_context", "query_complexity"],
            "model_selection": "gpt-4",
            "confidence_calculation": 0.85
        }
        
        message = Message(
            message_id="msg_003",
            conversation_id="conv_123",
            message_type=MessageType.ASSISTANT,
            content="Based on your question...",
            sender_type="ai",
            reasoning_trace=reasoning
        )
        
        assert message.reasoning_trace == reasoning
        assert "decision_factors" in message.reasoning_trace
    
    def test_message_with_quality_metrics(self):
        """Test message with quality metrics"""
        message = Message(
            message_id="msg_004",
            conversation_id="conv_123",
            message_type=MessageType.ASSISTANT,
            content="Here's a detailed response...",
            sender_type="ai",
            quality_score=0.88,
            validation_status="passed",
            flags=["high_quality", "comprehensive"]
        )
        
        assert message.quality_score == 0.88
        assert message.validation_status == "passed"
        assert "high_quality" in message.flags

class TestChatRequest:
    """Test ChatRequest model"""
    
    def test_chat_request_basic(self):
        """Test basic chat request"""
        request = ChatRequest(
            message="What is machine learning?",
            request_id="req_001"
        )
        
        assert request.message == "What is machine learning?"
        assert request.request_id == "req_001"
        assert request.response_mode == "intelligent"
        assert isinstance(request.timestamp, datetime)
    
    def test_chat_request_with_context(self):
        """Test chat request with context"""
        context = ChatContext(
            user_id="user_123",
            session_id="session_456"
        )
        
        request = ChatRequest(
            message="Continue our previous discussion",
            conversation_id="conv_789",
            preferred_model="claude",
            response_mode="creative",
            context=context,
            request_id="req_002"
        )
        
        assert request.conversation_id == "conv_789"
        assert request.preferred_model == "claude"
        assert request.response_mode == "creative"
        assert request.context.user_id == "user_123"
    
    def test_chat_request_with_additional_context(self):
        """Test chat request with additional context"""
        additional_context = {
            "document_refs": ["doc1", "doc2"],
            "previous_topics": ["AI", "machine learning"],
            "urgency": "high"
        }
        
        request = ChatRequest(
            message="Analyze these documents",
            additional_context=additional_context,
            request_id="req_003"
        )
        
        assert request.additional_context == additional_context
        assert "document_refs" in request.additional_context

class TestChatResponse:
    """Test ChatResponse model"""
    
    def test_chat_response_basic(self):
        """Test basic chat response"""
        response = ChatResponse(
            response="Machine learning is a subset of AI...",
            message_id="msg_response_001",
            conversation_id="conv_123",
            model_used="gpt-4",
            confidence=0.89,
            reasoning={"selection_reason": "best_for_technical_questions"},
            processing_time=2.3,
            tokens_used=200,
            cost_estimate=0.004
        )
        
        assert response.response.startswith("Machine learning")
        assert response.model_used == "gpt-4"
        assert response.confidence == 0.89
        assert response.processing_time == 2.3
        assert response.tokens_used == 200
        assert response.cost_estimate == 0.004
        assert isinstance(response.generated_at, datetime)
    
    def test_chat_response_with_metadata(self):
        """Test chat response with comprehensive metadata"""
        reasoning = {
            "model_selection": "claude",
            "confidence_calculation": "user_feedback_history",
            "fallback_chain": ["gpt-4", "gemini"]
        }
        
        decision_factors = ["query_complexity", "user_expertise", "domain_knowledge"]
        alternative_models = ["gpt-4", "gemini"]
        audit_trail = [
            {"step": "analysis", "result": "complex_query"},
            {"step": "model_selection", "result": "claude"}
        ]
        
        response = ChatResponse(
            response="Here's a comprehensive analysis...",
            message_id="msg_response_002",
            conversation_id="conv_123",
            model_used="claude",
            confidence=0.94,
            quality_score=0.92,
            safety_score=0.98,
            reasoning=reasoning,
            decision_factors=decision_factors,
            alternative_models=alternative_models,
            processing_time=3.1,
            tokens_used=350,
            cost_estimate=0.007,
            audit_trail=audit_trail
        )
        
        assert response.quality_score == 0.92
        assert response.safety_score == 0.98
        assert len(response.decision_factors) == 3
        assert len(response.alternative_models) == 2
        assert len(response.audit_trail) == 2
    
    def test_chat_response_with_compliance(self):
        """Test chat response with compliance information"""
        compliance_check = {
            "gdpr_compliant": True,
            "data_retention": "30_days",
            "privacy_level": "standard"
        }
        
        response = ChatResponse(
            response="Your data is handled securely...",
            message_id="msg_response_003",
            conversation_id="conv_123",
            model_used="gpt-4",
            confidence=0.91,
            reasoning={"compliance_verified": True},
            processing_time=1.8,
            tokens_used=180,
            cost_estimate=0.003,
            compliance_check=compliance_check
        )
        
        assert response.compliance_check["gdpr_compliant"]
        assert response.compliance_check["data_retention"] == "30_days"

class TestConversation:
    """Test Conversation model"""
    
    def test_conversation_creation(self):
        """Test creating a conversation"""
        context = ChatContext(
            user_id="user_123",
            session_id="session_456"
        )
        
        conversation = Conversation(
            conversation_id="conv_001",
            title="AI Discussion",
            context=context,
            status=ConversationStatus.ACTIVE
        )
        
        assert conversation.conversation_id == "conv_001"
        assert conversation.title == "AI Discussion"
        assert conversation.status == ConversationStatus.ACTIVE
        assert conversation.context.user_id == "user_123"
        assert conversation.message_count == 0
        assert len(conversation.messages) == 0
    
    def test_conversation_with_messages(self):
        """Test conversation with messages"""
        context = ChatContext(user_id="user_123", session_id="session_456")
        
        messages = [
            Message(
                message_id="msg_001",
                conversation_id="conv_001",
                message_type=MessageType.USER,
                content="Hello",
                sender_type="user"
            ),
            Message(
                message_id="msg_002",
                conversation_id="conv_001",
                message_type=MessageType.ASSISTANT,
                content="Hi there!",
                sender_type="ai",
                model_used="gpt-4"
            )
        ]
        
        conversation = Conversation(
            conversation_id="conv_001",
            context=context,
            messages=messages,
            message_count=2,
            models_used=["gpt-4"],
            total_tokens=50,
            total_cost=0.001
        )
        
        assert len(conversation.messages) == 2
        assert conversation.message_count == 2
        assert "gpt-4" in conversation.models_used
        assert conversation.total_tokens == 50
        assert conversation.total_cost == 0.001
    
    def test_conversation_with_analytics(self):
        """Test conversation with analytics data"""
        context = ChatContext(user_id="user_123", session_id="session_456")
        
        conversation = Conversation(
            conversation_id="conv_001",
            context=context,
            average_response_time=2.5,
            average_quality_score=0.88,
            user_satisfaction=0.92,
            conversation_rating=5,
            tags=["technical", "ai", "helpful"],
            key_topics=["machine learning", "neural networks"]
        )
        
        assert conversation.average_response_time == 2.5
        assert conversation.average_quality_score == 0.88
        assert conversation.user_satisfaction == 0.92
        assert conversation.conversation_rating == 5
        assert "technical" in conversation.tags
        assert "machine learning" in conversation.key_topics

class TestConversationSummary:
    """Test ConversationSummary model"""
    
    def test_conversation_summary_creation(self):
        """Test creating a conversation summary"""
        summary = ConversationSummary(
            conversation_id="conv_001",
            message_count=10,
            duration_minutes=25.5,
            user_messages=5,
            ai_messages=5,
            average_quality=0.86,
            average_confidence=0.89,
            user_satisfaction=0.91,
            average_response_time=2.1,
            total_tokens=1500,
            total_cost=0.030,
            main_topics=["AI", "programming", "best practices"],
            models_used=["gpt-4", "claude"],
            complexity_level="intermediate"
        )
        
        assert summary.message_count == 10
        assert summary.duration_minutes == 25.5
        assert summary.user_messages == 5
        assert summary.ai_messages == 5
        assert summary.average_quality == 0.86
        assert summary.total_cost == 0.030
        assert len(summary.main_topics) == 3
        assert len(summary.models_used) == 2
        assert summary.complexity_level == "intermediate"

class TestChatFeedback:
    """Test ChatFeedback model"""
    
    def test_chat_feedback_creation(self):
        """Test creating chat feedback"""
        feedback = ChatFeedback(
            feedback_id="feedback_001",
            message_id="msg_001",
            conversation_id="conv_001",
            user_id="user_123",
            rating=4,
            feedback_type="helpful",
            comment="Very informative response",
            accuracy_rating=5,
            helpfulness_rating=4,
            clarity_rating=4
        )
        
        assert feedback.feedback_id == "feedback_001"
        assert feedback.rating == 4
        assert feedback.feedback_type == "helpful"
        assert feedback.comment == "Very informative response"
        assert feedback.accuracy_rating == 5
        assert feedback.helpfulness_rating == 4
        assert feedback.clarity_rating == 4
        assert isinstance(feedback.created_at, datetime)
    
    def test_chat_feedback_with_issues(self):
        """Test chat feedback with reported issues"""
        feedback = ChatFeedback(
            feedback_id="feedback_002",
            message_id="msg_002",
            conversation_id="conv_001",
            user_id="user_123",
            rating=2,
            feedback_type="not_helpful",
            comment="Response was inaccurate",
            reported_issues=["factual_error", "incomplete_information"],
            improvement_suggestions="Please verify facts before responding"
        )
        
        assert feedback.rating == 2
        assert feedback.feedback_type == "not_helpful"
        assert len(feedback.reported_issues) == 2
        assert "factual_error" in feedback.reported_issues
        assert feedback.improvement_suggestions.startswith("Please verify")

class TestChatAnalytics:
    """Test ChatAnalytics model"""
    
    def test_chat_analytics_creation(self):
        """Test creating chat analytics"""
        analytics = ChatAnalytics(
            analytics_id="analytics_001",
            time_period="weekly",
            total_conversations=150,
            total_messages=750,
            unique_users=85,
            average_response_time=2.3,
            average_quality_score=0.87,
            user_satisfaction_rate=0.89,
            model_usage_stats={"gpt-4": 400, "claude": 250, "gemini": 100},
            total_tokens_used=125000,
            total_cost=25.50,
            cost_per_conversation=0.17
        )
        
        assert analytics.total_conversations == 150
        assert analytics.total_messages == 750
        assert analytics.unique_users == 85
        assert analytics.average_response_time == 2.3
        assert analytics.user_satisfaction_rate == 0.89
        assert analytics.model_usage_stats["gpt-4"] == 400
        assert analytics.total_cost == 25.50
        assert analytics.cost_per_conversation == 0.17
    
    def test_chat_analytics_with_trends(self):
        """Test chat analytics with trend data"""
        analytics = ChatAnalytics(
            analytics_id="analytics_002",
            time_period="monthly",
            total_conversations=600,
            total_messages=3000,
            unique_users=320,
            average_response_time=2.1,
            average_quality_score=0.89,
            user_satisfaction_rate=0.91,
            popular_topics=["AI", "programming", "data science", "web development"],
            peak_usage_hours=[9, 10, 14, 15, 16]
        )
        
        assert len(analytics.popular_topics) == 4
        assert "AI" in analytics.popular_topics
        assert len(analytics.peak_usage_hours) == 5
        assert 14 in analytics.peak_usage_hours

class TestChatUtilities:
    """Test utility functions"""
    
    def test_create_chat_context_utility(self):
        """Test create_chat_context utility function"""
        context = create_chat_context(
            user_id="user_123",
            session_id="session_456",
            conversation_id="conv_789",
            user_role="admin",
            platform="mobile"
        )
        
        assert isinstance(context, ChatContext)
        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.conversation_id == "conv_789"
        assert context.user_role == "admin"
        assert context.platform == "mobile"
    
    def test_create_message_utility(self):
        """Test create_message utility function"""
        message = create_message(
            conversation_id="conv_123",
            content="Test message",
            message_type=MessageType.USER,
            sender_type="user",
            sender_id="user_456"
        )
        
        assert isinstance(message, Message)
        assert message.conversation_id == "conv_123"
        assert message.content == "Test message"
        assert message.message_type == MessageType.USER
        assert message.sender_type == "user"
        assert message.sender_id == "user_456"
        assert message.message_id  # Should be generated
    
    def test_validate_chat_request_success(self):
        """Test successful chat request validation"""
        request = ChatRequest(
            message="This is a valid message",
            request_id="req_001"
        )
        
        is_valid, errors = validate_chat_request(request)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_chat_request_empty_message(self):
        """Test chat request validation with empty message"""
        request = ChatRequest(
            message="   ",  # Only whitespace
            request_id="req_002"
        )
        
        is_valid, errors = validate_chat_request(request)
        
        assert not is_valid
        assert "Message content cannot be empty" in errors
    
    def test_validate_chat_request_too_long(self):
        """Test chat request validation with message too long"""
        long_message = "x" * 10001  # Exceeds 10000 character limit
        
        request = ChatRequest(
            message=long_message,
            request_id="req_003"
        )
        
        is_valid, errors = validate_chat_request(request)
        
        assert not is_valid
        assert "Message too long" in errors[0]
    
    def test_validate_chat_request_missing_user_context(self):
        """Test chat request validation with missing user ID in context"""
        context = ChatContext(
            user_id="",  # Empty user ID
            session_id="session_123"
        )
        
        request = ChatRequest(
            message="Valid message",
            context=context,
            request_id="req_004"
        )
        
        is_valid, errors = validate_chat_request(request)
        
        assert not is_valid
        assert "User ID required in context" in errors
    
    def test_calculate_conversation_metrics(self):
        """Test conversation metrics calculation"""
        messages = [
            Message(
                message_id="msg_001",
                conversation_id="conv_001",
                message_type=MessageType.ASSISTANT,
                content="Response 1",
                sender_type="ai",
                confidence_score=0.85,
                quality_score=0.88,
                processing_time=2.1,
                tokens_used=100,
                cost_estimate=0.002
            ),
            Message(
                message_id="msg_002", 
                conversation_id="conv_001",
                message_type=MessageType.ASSISTANT,
                content="Response 2",
                sender_type="ai",
                confidence_score=0.92,
                quality_score=0.90,
                processing_time=1.8,
                tokens_used=150,
                cost_estimate=0.003
            )
        ]
        
        conversation = Conversation(
            conversation_id="conv_001",
            context=ChatContext(user_id="user_123", session_id="session_456"),
            messages=messages
        )
        
        metrics = calculate_conversation_metrics(conversation)
        
        assert "average_confidence" in metrics
        assert "average_quality" in metrics
        assert "average_response_time" in metrics
        assert "total_tokens" in metrics
        assert "total_cost" in metrics
        
        assert metrics["average_confidence"] == 0.885  # (0.85 + 0.92) / 2
        assert metrics["average_quality"] == 0.89      # (0.88 + 0.90) / 2
        assert metrics["total_tokens"] == 250          # 100 + 150
        assert metrics["total_cost"] == 0.005          # 0.002 + 0.003

class TestChatIntegration:
    """Integration tests for chat functionality"""
    
    def test_complete_chat_flow(self):
        """Test complete chat interaction flow"""
        # 1. Create chat context
        context = create_chat_context(
            user_id="integration_user",
            session_id="integration_session",
            user_role="developer",
            platform="web"
        )
        
        # 2. Create chat request
        request = ChatRequest(
            message="Explain machine learning in simple terms",
            context=context,
            preferred_model="gpt-4",
            response_mode="educational",
            request_id="integration_req_001"
        )
        
        # 3. Validate request
        is_valid, errors = validate_chat_request(request)
        assert is_valid
        
        # 4. Create conversation
        conversation = Conversation(
            conversation_id="integration_conv_001",
            title="Machine Learning Discussion",
            context=context,
            status=ConversationStatus.ACTIVE
        )
        
        # 5. Create user message
        user_message = create_message(
            conversation_id=conversation.conversation_id,
            content=request.message,
            message_type=MessageType.USER,
            sender_type="user",
            sender_id=context.user_id
        )
        
        # 6. Create AI response
        ai_response = ChatResponse(
            response="Machine learning is a way for computers to learn patterns...",
            message_id="integration_response_001",
            conversation_id=conversation.conversation_id,
            model_used="gpt-4",
            confidence=0.91,
            quality_score=0.89,
            reasoning={"educational_mode": True, "simplified_language": True},
            processing_time=2.3,
            tokens_used=200,
            cost_estimate=0.004
        )
        
        # 7. Create AI message from response
        ai_message = create_message(
            conversation_id=conversation.conversation_id,
            content=ai_response.response,
            message_type=MessageType.ASSISTANT,
            sender_type="ai",
            model_used=ai_response.model_used,
            confidence_score=ai_response.confidence,
            quality_score=ai_response.quality_score,
            processing_time=ai_response.processing_time,
            tokens_used=ai_response.tokens_used,
            cost_estimate=ai_response.cost_estimate
        )
        
        # 8. Update conversation with messages
        conversation.messages = [user_message, ai_message]
        conversation.message_count = 2
        conversation.models_used = [ai_response.model_used]
        
        # 9. Calculate metrics
        metrics = calculate_conversation_metrics(conversation)
        conversation.average_response_time = metrics.get("average_response_time", 0)
        conversation.total_tokens = metrics.get("total_tokens", 0)
        conversation.total_cost = metrics.get("total_cost", 0)
        
        # 10. Create feedback
        feedback = ChatFeedback(
            feedback_id="integration_feedback_001",
            message_id=ai_message.message_id,
            conversation_id=conversation.conversation_id,
            user_id=context.user_id,
            rating=5,
            feedback_type="helpful",
            comment="Great explanation!",
            helpfulness_rating=5,
            clarity_rating=5
        )
        
        # Verify the complete flow
        assert conversation.conversation_id == "integration_conv_001"
        assert len(conversation.messages) == 2
        assert conversation.messages[0].message_type == MessageType.USER
        assert conversation.messages[1].message_type == MessageType.ASSISTANT
        assert conversation.total_tokens == 200
        assert feedback.rating == 5
    
    def test_conversation_summary_generation(self):
        """Test conversation summary generation"""
        # Create a conversation with multiple messages
        context = ChatContext(user_id="summary_user", session_id="summary_session")
        
        messages = []
        total_tokens = 0
        total_cost = 0.0
        
        # Create alternating user and AI messages
        for i in range(6):  # 3 user messages, 3 AI messages
            if i % 2 == 0:  # User message
                message = create_message(
                    conversation_id="summary_conv",
                    content=f"User question {i//2 + 1}",
                    message_type=MessageType.USER,
                    sender_type="user"
                )
            else:  # AI message
                tokens = 150 + (i * 20)
                cost = tokens * 0.00002
                total_tokens += tokens
                total_cost += cost
                
                message = create_message(
                    conversation_id="summary_conv",
                    content=f"AI response {i//2 + 1}",
                    message_type=MessageType.ASSISTANT,
                    sender_type="ai",
                    model_used="gpt-4",
                    confidence_score=0.85 + (i * 0.02),
                    quality_score=0.88 + (i * 0.01),
                    processing_time=2.0 + (i * 0.1),
                    tokens_used=tokens,
                    cost_estimate=cost
                )
            
            messages.append(message)
        
        conversation = Conversation(
            conversation_id="summary_conv",
            context=context,
            messages=messages,
            message_count=len(messages),
            models_used=["gpt-4"],
            total_tokens=total_tokens,
            total_cost=total_cost
        )
        
        # Create conversation summary
        summary = ConversationSummary(
            conversation_id=conversation.conversation_id,
            message_count=conversation.message_count,
            duration_minutes=15.5,
            user_messages=3,
            ai_messages=3,
            average_quality=0.89,
            average_confidence=0.87,
            user_satisfaction=0.92,
            average_response_time=2.25,
            total_tokens=conversation.total_tokens,
            total_cost=conversation.total_cost,
            main_topics=["AI", "programming"],
            models_used=conversation.models_used,
            complexity_level="intermediate"
        )
        
        assert summary.message_count == 6
        assert summary.user_messages == 3
        assert summary.ai_messages == 3
        assert summary.total_tokens == total_tokens
        assert summary.total_cost == total_cost
        assert "AI" in summary.main_topics

class TestChatErrorHandling:
    """Test error handling in chat functionality"""
    
    def test_invalid_message_types(self):
        """Test handling of invalid message types"""
        # This would be caught by Pydantic validation
        with pytest.raises(ValueError):
            Message(
                message_id="invalid_msg",
                conversation_id="conv_001",
                message_type="invalid_type",  # Not a valid MessageType
                content="Invalid message",
                sender_type="user"
            )
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        with pytest.raises(ValueError):
            ChatRequest(
                # Missing required 'message' field
                request_id="req_001"
            )
    
    def test_invalid_ratings(self):
        """Test handling of invalid feedback ratings"""
        with pytest.raises(ValueError):
            ChatFeedback(
                feedback_id="feedback_001",
                message_id="msg_001",
                conversation_id="conv_001",
                user_id="user_123",
                rating=6,  # Invalid rating > 5
                feedback_type="helpful"
            )

class TestChatPerformance:
    """Performance tests for chat functionality"""
    
    def test_large_conversation_handling(self):
        """Test handling of conversations with many messages"""
        context = ChatContext(user_id="perf_user", session_id="perf_session")
        
        # Create conversation with many messages
        messages = []
        for i in range(100):
            message_type = MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
            sender_type = "user" if i % 2 == 0 else "ai"
            
            message = create_message(
                conversation_id="perf_conv",
                content=f"Message content {i}",
                message_type=message_type,
                sender_type=sender_type
            )
            messages.append(message)
        
        conversation = Conversation(
            conversation_id="perf_conv",
            context=context,
            messages=messages,
            message_count=len(messages)
        )
        
        # Calculate metrics - should complete quickly
        import time
        start_time = time.time()
        metrics = calculate_conversation_metrics(conversation)
        calculation_time = time.time() - start_time
        
        assert calculation_time < 1.0  # Should complete within 1 second
        assert len(conversation.messages) == 100

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
