# backend/services/01-core/nava-logic-controller/app/core/context_manager.py
"""
Context Manager - Session and Conversation Context Management
Handles conversation history, user preferences, and context continuity
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from uuid import uuid4

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Individual message in conversation"""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    user_id: Optional[str]
    started_at: datetime
    last_active: datetime
    messages: List[Message]
    user_preferences: Dict[str, Any]
    context_summary: str
    total_messages: int
    session_metadata: Dict[str, Any]

class ContextManager:
    """
    Manages conversation context and session state
    Handles message history, user preferences, and context continuity
    """
    
    def __init__(self, max_context_length: int = 4000, max_messages: int = 50):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_context_length = max_context_length
        self.max_messages = max_messages
        self.session_timeout = timedelta(hours=2)  # 2 hours timeout
        
        self.default_preferences = {
            "preferred_ai_model": None,
            "response_style": "balanced",  # concise, detailed, balanced
            "language": "en",
            "technical_level": "intermediate",  # beginner, intermediate, advanced
            "content_filtering": "standard",  # strict, standard, relaxed
            "conversation_mode": "assistant"  # assistant, collaborative, tutoring
        }
        
        logger.info("✅ Context Manager initialized")
    
    def create_session(self, user_id: Optional[str] = None, 
                      initial_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid4())
        
        preferences = self.default_preferences.copy()
        if initial_preferences:
            preferences.update(initial_preferences)
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            last_active=datetime.now(),
            messages=[],
            user_preferences=preferences,
            context_summary="",
            total_messages=0,
            session_metadata={}
        )
        
        self.sessions[session_id] = session
        logger.info(f"✅ Created session {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check if session has expired
            if datetime.now() - session.last_active > self.session_timeout:
                logger.warning(f"⚠️ Session {session_id[:8]}... expired")
                self.cleanup_session(session_id)
                return None
            
            return session
        return None
    
    def add_message(self, session_id: str, role: str, content: str,
                   model_used: Optional[str] = None,
                   processing_time: Optional[float] = None,
                   quality_score: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to the conversation"""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"❌ Session {session_id[:8]}... not found")
            return False
        
        message = Message(
            id=str(uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            model_used=model_used,
            processing_time=processing_time,
            quality_score=quality_score,
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.total_messages += 1
        session.last_active = datetime.now()
        
        # Maintain message limit
        if len(session.messages) > self.max_messages:
            session.messages = session.messages[-self.max_messages:]
        
        # Update context summary if needed
        self._update_context_summary(session)
        
        logger.debug(f"✅ Added {role} message to session {session_id[:8]}...")
        return True
    
    def get_conversation_context(self, session_id: str, 
                               max_messages: Optional[int] = None) -> Dict[str, Any]:
        """Get conversation context for AI model"""
        session = self.get_session(session_id)
        if not session:
            return self._get_empty_context()
        
        # Determine message limit
        if max_messages is None:
            max_messages = min(20, len(session.messages))
        
        # Get recent messages
        recent_messages = session.messages[-max_messages:] if session.messages else []
        
        # Convert messages to API format
        formatted_messages = []
        for msg in recent_messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        # Calculate context length
        context_text = " ".join([msg.content for msg in recent_messages])
        context_length = len(context_text)
        
        # Truncate if too long
        if context_length > self.max_context_length:
            formatted_messages = self._truncate_context(formatted_messages)
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "user_preferences": session.user_preferences,
            "context_summary": session.context_summary,
            "total_messages": session.total_messages,
            "session_duration_minutes": int((datetime.now() - session.started_at).total_seconds() / 60),
            "last_active": session.last_active.isoformat(),
            "context_metadata": {
                "message_count": len(formatted_messages),
                "context_length": len(" ".join([msg["content"] for msg in formatted_messages])),
                "session_started": session.started_at.isoformat()
            }
        }
    
    def update_user_preferences(self, session_id: str, 
                              preferences: Dict[str, Any]) -> bool:
        """Update user preferences for the session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.user_preferences.update(preferences)
        session.last_active = datetime.now()
        
        logger.info(f"✅ Updated preferences for session {session_id[:8]}...")
        return True
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences for the session"""
        session = self.get_session(session_id)
        if not session:
            return self.default_preferences.copy()
        
        return session.user_preferences.copy()
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get conversation summary"""
        session = self.get_session(session_id)
        if not session:
            return ""
        
        if session.context_summary:
            return session.context_summary
        
        # Generate summary from recent messages
        if session.messages:
            recent_messages = session.messages[-5:]  # Last 5 messages
            summary_parts = []
            
            for msg in recent_messages:
                if msg.role == "user":
                    summary_parts.append(f"User asked: {msg.content[:100]}...")
                elif msg.role == "assistant":
                    summary_parts.append(f"Assistant responded about: {msg.content[:100]}...")
            
            return " ".join(summary_parts)
        
        return "New conversation"
    
    def search_conversation_history(self, session_id: str, 
                                  query: str, 
                                  max_results: int = 5) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for msg in session.messages:
            if query_lower in msg.content.lower():
                matches.append({
                    "message_id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "relevance_score": self._calculate_relevance(query, msg.content)
                })
        
        # Sort by relevance and timestamp
        matches.sort(key=lambda x: (x["relevance_score"], x["timestamp"]), reverse=True)
        
        return matches[:max_results]
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for the session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Calculate metrics
        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]
        
        total_processing_time = sum(
            msg.processing_time for msg in assistant_messages 
            if msg.processing_time is not None
        )
        
        avg_processing_time = (
            total_processing_time / len(assistant_messages) 
            if assistant_messages else 0
        )
        
        quality_scores = [
            msg.quality_score for msg in assistant_messages 
            if msg.quality_score is not None
        ]
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Model usage
        model_usage = {}
        for msg in assistant_messages:
            if msg.model_used:
                model_usage[msg.model_used] = model_usage.get(msg.model_used, 0) + 1
        
        return {
            "session_id": session_id,
            "total_messages": session.total_messages,
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "session_duration_minutes": int((datetime.now() - session.started_at).total_seconds() / 60),
            "avg_processing_time_seconds": round(avg_processing_time, 2),
            "avg_quality_score": round(avg_quality_score, 3),
            "model_usage": model_usage,
            "last_active": session.last_active.isoformat(),
            "started_at": session.started_at.isoformat()
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_active > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        if expired_sessions:
            logger.info(f"✅ Cleaned up {len(expired_sessions)} expired sessions")
    
    def cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"✅ Cleaned up session {session_id[:8]}...")
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Convert to serializable format
        export_data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "started_at": session.started_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "user_preferences": session.user_preferences,
            "context_summary": session.context_summary,
            "total_messages": session.total_messages,
            "session_metadata": session.session_metadata,
            "messages": []
        }
        
        for msg in session.messages:
            export_data["messages"].append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "model_used": msg.model_used,
                "processing_time": msg.processing_time,
                "quality_score": msg.quality_score,
                "metadata": msg.metadata
            })
        
        return export_data
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)
    
    def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active sessions"""
        summaries = []
        
        for session_id, session in self.sessions.items():
            summaries.append({
                "session_id": session_id,
                "user_id": session.user_id,
                "started_at": session.started_at.isoformat(),
                "last_active": session.last_active.isoformat(),
                "total_messages": session.total_messages,
                "duration_minutes": int((datetime.now() - session.started_at).total_seconds() / 60),
                "is_active": (datetime.now() - session.last_active).total_seconds() < 300  # 5 minutes
            })
        
        return summaries
    
    def _update_context_summary(self, session: ConversationSession):
        """Update context summary based on conversation"""
        if len(session.messages) % 10 == 0:  # Update every 10 messages
            recent_messages = session.messages[-10:]
            
            # Simple summary generation
            topics = []
            for msg in recent_messages:
                if msg.role == "user" and len(msg.content) > 20:
                    # Extract key topics (simple keyword extraction)
                    words = msg.content.lower().split()
                    key_words = [w for w in words if len(w) > 5][:3]
                    topics.extend(key_words)
            
            if topics:
                session.context_summary = f"Discussed: {', '.join(set(topics))}"
    
    def _truncate_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate context to fit within limits"""
        total_length = 0
        truncated_messages = []
        
        # Work backwards from most recent
        for msg in reversed(messages):
            msg_length = len(msg["content"])
            if total_length + msg_length <= self.max_context_length:
                truncated_messages.insert(0, msg)
                total_length += msg_length
            else:
                break
        
        return truncated_messages
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score for search"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def _get_empty_context(self) -> Dict[str, Any]:
        """Get empty context for new sessions"""
        return {
            "session_id": None,
            "messages": [],
            "user_preferences": self.default_preferences.copy(),
            "context_summary": "",
            "total_messages": 0,
            "session_duration_minutes": 0,
            "last_active": datetime.now().isoformat(),
            "context_metadata": {
                "message_count": 0,
                "context_length": 0,
                "session_started": datetime.now().isoformat()
            }
        }

# Global instance
context_manager = ContextManager()

def create_conversation_session(user_id: Optional[str] = None, 
                               preferences: Optional[Dict[str, Any]] = None) -> str:
    """Create a new conversation session"""
    return context_manager.create_session(user_id, preferences)

def add_conversation_message(session_id: str, role: str, content: str,
                            model_used: Optional[str] = None,
                            processing_time: Optional[float] = None,
                            quality_score: Optional[float] = None) -> bool:
    """Add a message to conversation"""
    return context_manager.add_message(
        session_id, role, content, model_used, processing_time, quality_score
    )

def get_context_for_ai(session_id: str, max_messages: int = 20) -> Dict[str, Any]:
    """Get conversation context for AI model"""
    return context_manager.get_conversation_context(session_id, max_messages)

def update_session_preferences(session_id: str, preferences: Dict[str, Any]) -> bool:
    """Update user preferences"""
    return context_manager.update_user_preferences(session_id, preferences)

def get_session_preferences(session_id: str) -> Dict[str, Any]:
    """Get user preferences"""
    return context_manager.get_user_preferences(session_id)

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    context_manager.cleanup_expired_sessions()
            