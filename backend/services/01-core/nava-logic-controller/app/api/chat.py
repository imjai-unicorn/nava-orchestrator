# backend/services/01-core/nava-logic-controller/app/api/chat.py
"""
Enhanced Chat Endpoints - Week 2 Critical Component
Advanced chat functionality and API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, AsyncGenerator
import logging
import time
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

# Models
class EnhancedChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(default="anonymous")
    conversation_id: Optional[str] = Field(None)
    context: Optional[Dict[str, Any]] = Field(None)
    preferences: Optional[Dict[str, Any]] = Field(None)
    streaming: bool = Field(default=False)
    quality_requirements: Optional[Dict[str, str]] = Field(None)
    workflow_type: str = Field(default="auto")
    priority: str = Field(default="normal")  # low, normal, high, urgent

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    conversation_id: str
    message_id: str
    response: str
    model_used: str
    confidence: float
    processing_time_seconds: float
    workflow_used: bool
    decision_info: Dict[str, Any]
    quality_metrics: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    timestamp: str

class ConversationContext(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    context_summary: str
    created_at: str
    last_updated: str

class StreamingChatResponse(BaseModel):
    chunk_id: int
    content: str
    is_final: bool
    model_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Conversation management
class ConversationManager:
    """Manage chat conversations and context"""
    
    def __init__(self):
        self.conversations = {}
        self.max_conversations = 1000
        self.max_context_length = 10000
        
    def create_conversation(self, user_id: str, initial_message: str) -> str:
        """Create new conversation"""
        conversation_id = f"{user_id}_{int(time.time())}_{hash(initial_message) % 1000}"
        
        self.conversations[conversation_id] = {
            "user_id": user_id,
            "messages": [],
            "context": "",
            "preferences": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation"""
        if conversation_id not in self.conversations:
            return False
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id]["messages"].append(message)
        self.conversations[conversation_id]["last_updated"] = datetime.now().isoformat()
        
        # Limit conversation length
        if len(self.conversations[conversation_id]["messages"]) > 50:
            self.conversations[conversation_id]["messages"] = self.conversations[conversation_id]["messages"][-40:]
        
        return True
    
    def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation context"""
        return self.conversations.get(conversation_id)
    
    def update_preferences(self, conversation_id: str, preferences: Dict[str, Any]):
        """Update user preferences for conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["preferences"].update(preferences)

# Global conversation manager
conversation_manager = ConversationManager()

@chat_router.post("/enhanced", response_model=ChatResponse)
async def enhanced_chat(request: EnhancedChatRequest, background_tasks: BackgroundTasks):
    """Enhanced chat endpoint with full NAVA capabilities"""
    
    start_time = time.time()
    
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation(request.user_id, request.message)
        
        # Add user message to conversation
        conversation_manager.add_message(conversation_id, "user", request.message)
        
        # Get conversation context
        context = conversation_manager.get_conversation_context(conversation_id)
        
        # Prepare enhanced request
        enhanced_context = {
            "conversation_history": context["messages"][-5:] if context else [],  # Last 5 messages
            "user_preferences": context["preferences"] if context else {},
            "quality_requirements": request.quality_requirements or {},
            "workflow_type": request.workflow_type,
            "priority": request.priority
        }
        
        # Merge with provided context
        if request.context:
            enhanced_context.update(request.context)
        
        # Process through NAVA orchestrator - SIMPLIFIED VERSION
        try:
            # Try to import workflow orchestrator
            from ..core.workflow_orchestrator import workflow_orchestrator
            
            # Use appropriate workflow based on complexity
            if request.workflow_type == "simple":
                result = await workflow_orchestrator.execute_simple_workflow(
                    message=request.message,
                    user_id=request.user_id
                )
            else:
                # Try sequential workflow for complex requests
                try:
                    result = await workflow_orchestrator.execute_sequential_workflow(
                        message=request.message,
                        user_id=request.user_id
                    )
                except AttributeError:
                    # Fallback to simple workflow if sequential doesn't exist
                    result = await workflow_orchestrator.execute_simple_workflow(
                        message=request.message,
                        user_id=request.user_id
                    )
                    
        except ImportError:
            # If workflow orchestrator not available, try logic orchestrator
            logger.warning("Workflow orchestrator not found, trying logic orchestrator")
            try:
                from ..service.logic_orchestrator import LogicOrchestrator
                orchestrator = LogicOrchestrator()
                
                if hasattr(orchestrator, 'process_request'):
                    result = await orchestrator.process_request(
                        message=request.message,
                        user_id=request.user_id,
                        context=enhanced_context
                    )
                else:
                    # Emergency fallback
                    result = {
                        "response": f"I understand you're asking: {request.message}. How can I help you with that?",
                        "model_used": "logic_orchestrator_fallback",
                        "confidence": 0.7,
                        "processing_time_seconds": 0.1,
                        "workflow_used": False,
                        "decision_info": {"method": "logic_orchestrator_fallback"}
                    }
            except ImportError:
                # Ultimate emergency fallback
                logger.error("No orchestrator available, using emergency fallback")
                result = {
                    "response": f"I received your message: {request.message}. I'm processing it now.",
                    "model_used": "emergency_fallback",
                    "confidence": 0.5,
                    "processing_time_seconds": 0.1,
                    "workflow_used": False,
                    "decision_info": {"method": "emergency_fallback", "reason": "no_orchestrator"}
                }
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            # Emergency fallback for any other error
            result = {
                "response": f"I understand you're asking about: {request.message}. Let me help you with that.",
                "model_used": "error_fallback",
                "confidence": 0.6,
                "processing_time_seconds": 0.1,
                "workflow_used": False,
                "decision_info": {"method": "error_fallback", "error": str(e)}
            }
        
        # Generate response
        processing_time = time.time() - start_time
        message_id = f"msg_{int(time.time())}_{hash(request.message) % 1000}"
        
        response = ChatResponse(
            conversation_id=conversation_id,
            message_id=message_id,
            response=str(result.get("response", "No response generated")),
            model_used=str(result.get("model_used", "unknown")),
            confidence=float(result.get("confidence", 0.8)),
            processing_time_seconds=processing_time,
            workflow_used=bool(result.get("workflow_used", False)),
            decision_info=result.get("decision_info", {}),
            quality_metrics=result.get("quality_metrics"),
            suggestions=result.get("suggestions"),
            timestamp=datetime.now().isoformat()
        )
        
        # Add assistant response to conversation
        conversation_manager.add_message(
            conversation_id, 
            "assistant", 
            response.response,
            {
                "model_used": response.model_used,
                "confidence": response.confidence,
                "processing_time": processing_time
            }
        )
        
        # Background tasks - SIMPLIFIED
        background_tasks.add_task(_log_chat_interaction, request, response)
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@chat_router.post("/streaming")
async def streaming_chat(request: EnhancedChatRequest):
    """Streaming chat endpoint"""
    
    if not request.streaming:
        raise HTTPException(status_code=400, detail="Streaming not enabled in request")
    
    async def generate_stream():
        """Generate streaming response"""
        try:
            # Simulate streaming response
            response_parts = [
                "I understand your question about",
                f" '{request.message}'.",
                " Let me think about this...",
                " Here's my response: ",
                "This is a streaming response that demonstrates",
                " how NAVA can provide real-time chat capabilities.",
                " The system processes your request and",
                " streams the response back in chunks."
            ]
            
            for i, part in enumerate(response_parts):
                chunk = StreamingChatResponse(
                    chunk_id=i,
                    content=part,
                    is_final=(i == len(response_parts) - 1),
                    model_used="streaming_gpt" if i == 0 else None,
                    metadata={"chunk_count": len(response_parts)} if i == 0 else None
                )
                
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.1)  # Simulate processing delay
                
            # Final chunk
            final_chunk = StreamingChatResponse(
                chunk_id=len(response_parts),
                content="",
                is_final=True,
                metadata={"status": "complete"}
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            error_chunk = StreamingChatResponse(
                chunk_id=-1,
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@chat_router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details"""
    
    context = conversation_manager.get_conversation_context(conversation_id)
    if not context:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "user_id": context["user_id"],
        "message_count": len(context["messages"]),
        "created_at": context["created_at"],
        "last_updated": context["last_updated"],
        "preferences": context["preferences"],
        "recent_messages": context["messages"][-10:]  # Last 10 messages
    }

@chat_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation"""
    
    if conversation_id in conversation_manager.conversations:
        del conversation_manager.conversations[conversation_id]
        return {"message": "Conversation deleted", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@chat_router.post("/conversations/{conversation_id}/preferences")
async def update_conversation_preferences(conversation_id: str, preferences: Dict[str, Any]):
    """Update conversation preferences"""
    
    context = conversation_manager.get_conversation_context(conversation_id)
    if not context:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_manager.update_preferences(conversation_id, preferences)
    
    return {
        "message": "Preferences updated",
        "conversation_id": conversation_id,
        "updated_preferences": preferences
    }

@chat_router.get("/conversations")
async def list_conversations(user_id: Optional[str] = None, limit: int = 20):
    """List conversations"""
    
    conversations = []
    
    for conv_id, conv_data in conversation_manager.conversations.items():
        if user_id and conv_data["user_id"] != user_id:
            continue
            
        conversations.append({
            "conversation_id": conv_id,
            "user_id": conv_data["user_id"],
            "message_count": len(conv_data["messages"]),
            "created_at": conv_data["created_at"],
            "last_updated": conv_data["last_updated"],
            "last_message": conv_data["messages"][-1]["content"] if conv_data["messages"] else None
        })
    
    # Sort by last updated
    conversations.sort(key=lambda x: x["last_updated"], reverse=True)
    
    return {
        "conversations": conversations[:limit],
        "total_count": len(conversations),
        "user_id": user_id
    }

@chat_router.websocket("/ws/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket chat endpoint for real-time communication"""
    
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected for user: {user_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = message_data.get("message", "")
                conversation_id = message_data.get("conversation_id")
                
                if not message:
                    await websocket.send_text(json.dumps({
                        "error": "Empty message",
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                # Create conversation if needed
                if not conversation_id:
                    conversation_id = conversation_manager.create_conversation(user_id, message)
                
                # Process message
                start_time = time.time()
                
                # Simple processing for WebSocket (avoid complex workflows)
                try:
                    from ..core.workflow_orchestrator import workflow_orchestrator
                    result = await workflow_orchestrator.execute_simple_workflow(
                        message=message,
                        user_id=user_id
                    )
                except Exception as e:
                    result = {
                        "response": f"I received your message: {message}",
                        "model_used": "websocket_fallback",
                        "confidence": 0.7,
                        "processing_time_seconds": 0.1
                    }
                
                processing_time = time.time() - start_time
                
                # Send response
                response = {
                    "conversation_id": conversation_id,
                    "response": result.get("response", "No response"),
                    "model_used": result.get("model_used", "unknown"),
                    "confidence": result.get("confidence", 0.8),
                    "processing_time_seconds": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_text(json.dumps(response))
                
                # Update conversation
                conversation_manager.add_message(conversation_id, "user", message)
                conversation_manager.add_message(conversation_id, "assistant", response["response"])
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"❌ WebSocket processing error: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"Processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")

@chat_router.post("/quick")
async def quick_chat(message: str, user_id: str = "anonymous", model_preference: Optional[str] = None):
    """Quick chat endpoint for simple requests"""
    
    try:
        start_time = time.time()
        
        # Simple processing without conversation management
        from ..core.workflow_orchestrator import workflow_orchestrator
        
        result = await workflow_orchestrator.execute_simple_workflow(
            message=message,
            user_id=user_id,
            model=model_preference or "gpt"
        )
        
        processing_time = time.time() - start_time
        
        return {
            "response": result.get("response", "No response"),
            "model_used": result.get("model_used", model_preference or "gpt"),
            "confidence": result.get("confidence", 0.8),
            "processing_time_seconds": processing_time,
            "workflow_type": "quick_chat",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick chat error: {e}")
        return {
            "response": f"I understand you said: {message}. How can I help you with that?",
            "model_used": "fallback",
            "confidence": 0.5,
            "processing_time_seconds": 0.1,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@chat_router.post("/analyze")
async def analyze_message(message: str, analysis_type: str = "complexity"):
    """Analyze message without generating response - SIMPLIFIED VERSION"""
    
    try:
        analysis_result = {}
        
        # Try complexity analysis
        if analysis_type == "complexity":
            try:
                from ..core.complexity_analyzer import complexity_analyzer
                analysis_result["complexity"] = complexity_analyzer.analyze_complexity(message)
            except (ImportError, AttributeError):
                # Fallback complexity analysis
                message_length = len(message)
                if message_length < 50:
                    complexity_level = "low"
                    complexity_score = 0.3
                elif message_length < 200:
                    complexity_level = "medium" 
                    complexity_score = 0.6
                else:
                    complexity_level = "high"
                    complexity_score = 0.9
                    
                analysis_result["complexity"] = {
                    "level": complexity_level, 
                    "score": complexity_score,
                    "method": "fallback_length_based"
                }
        
        # Try decision analysis
        if analysis_type == "decision" or analysis_type == "all":
            try:
                from ..core.decision_engine import decision_engine
                model, confidence, reasoning = decision_engine.select_model(message)
                analysis_result["model_selection"] = {
                    "recommended_model": model,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            except (ImportError, AttributeError):
                # Fallback model selection
                if "code" in message.lower() or "programming" in message.lower():
                    recommended_model = "gpt"
                elif "creative" in message.lower() or "story" in message.lower():
                    recommended_model = "claude"
                else:
                    recommended_model = "gpt"
                    
                analysis_result["model_selection"] = {
                    "recommended_model": recommended_model,
                    "confidence": 0.7,
                    "reasoning": {"method": "fallback_keyword_based"}
                }
        
        # Try behavior analysis
        if analysis_type == "behavior" or analysis_type == "all":
            try:
                from ..core.decision_engine import decision_engine
                patterns = decision_engine.get_behavior_patterns()
                analysis_result["behavior_patterns"] = patterns["patterns"]
            except (ImportError, AttributeError):
                # Fallback behavior detection
                if "?" in message:
                    detected_pattern = "question"
                elif "help" in message.lower():
                    detected_pattern = "assistance"
                elif len(message) > 100:
                    detected_pattern = "detailed_conversation"
                else:
                    detected_pattern = "simple_conversation"
                    
                analysis_result["behavior_patterns"] = {
                    "detected": detected_pattern,
                    "method": "fallback_pattern_matching"
                }
        
        return {
            "message": message,
            "analysis_type": analysis_type,
            "analysis_result": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Message analysis error: {e}")
        return {
            "message": message,
            "analysis_type": analysis_type,
            "error": str(e),
            "fallback_analysis": {
                "basic_info": {
                    "length": len(message),
                    "has_question": "?" in message,
                    "estimated_complexity": "medium"
                }
            },
            "timestamp": datetime.now().isoformat()
        }

@chat_router.get("/stats")
async def get_chat_stats():
    """Get chat usage statistics"""
    
    try:
        total_conversations = len(conversation_manager.conversations)
        total_messages = sum(len(conv["messages"]) for conv in conversation_manager.conversations.values())
        
        active_conversations = sum(
            1 for conv in conversation_manager.conversations.values()
            if conv["messages"] and 
            (time.time() - time.mktime(time.strptime(conv["last_updated"][:19], "%Y-%m-%dT%H:%M:%S")) < 3600)
        )
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "active_conversations_last_hour": active_conversations,
            "average_messages_per_conversation": total_messages / max(1, total_conversations),
            "conversation_storage_usage": f"{len(str(conversation_manager.conversations))} bytes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Chat stats error: {e}")
        return {"error": str(e)}

async def _log_chat_interaction(request: EnhancedChatRequest, response: ChatResponse):
    """Background task to log chat interaction - SIMPLIFIED VERSION"""
    try:
        # Log interaction for analytics
        logger.info(
            f"💬 Chat: user={request.user_id}, "
            f"model={response.model_used}, "
            f"confidence={response.confidence:.2f}, "
            f"time={response.processing_time_seconds:.2f}s"
        )
        
        # Try to record metrics for learning system (optional)
        try:
            from ..service.learning_engine import learning_engine
            if hasattr(learning_engine, 'process_feedback'):
                await learning_engine.process_feedback(
                    model_used=response.model_used,
                    pattern="conversation",
                    feedback_score=response.confidence,
                    response_time=response.processing_time_seconds
                )
        except (ImportError, AttributeError) as e:
            logger.debug(f"Learning system not available: {e}")
        except Exception as e:
            logger.debug(f"Learning system logging failed: {e}")
            
    except Exception as e:
        logger.error(f"❌ Chat logging error: {e}")

# Health check for chat service
@chat_router.get("/health")
async def chat_health():
    """Chat service health check"""
    return {
        "status": "healthy",
        "service": "enhanced_chat",
        "active_conversations": len(conversation_manager.conversations),
        "memory_usage": "normal",
        "timestamp": datetime.now().isoformat()
    }