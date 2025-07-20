# backend/services/06-enterprise-security/auth-service/app/session_manager.py
"""
Session Manager
Enterprise session management and security monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import secrets
import hashlib
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Pydantic models
class SessionCreate(BaseModel):
    user_id: str
    username: str
    role: str
    device_info: Optional[str] = None
    ip_address: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    username: str
    role: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str]
    device_info: Optional[str]
    is_active: bool

class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total_count: int
    active_count: int

class SessionSecurityEvent(BaseModel):
    event_type: str
    session_id: str
    user_id: str
    timestamp: datetime
    details: Dict[str, Any]

class SessionManager:
    def __init__(self):
        self.sessions = {}  # In production, use Redis or database
        self.security_events = []
        self.max_sessions_per_user = 5
        self.session_timeout = timedelta(hours=8)  # 8 hours
        self.security_timeout = timedelta(minutes=30)  # 30 min for sensitive operations
        logger.info("🔒 Session Manager initialized")

    def generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)

    def create_session(self, user_id: str, username: str, role: str, 
                      ip_address: Optional[str] = None, 
                      device_info: Optional[str] = None) -> str:
        """Create new session"""
        
        # Check existing sessions for user
        user_sessions = [s for s in self.sessions.values() 
                        if s["user_id"] == user_id and s["is_active"]]
        
        # Limit concurrent sessions
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda x: x["created_at"])
            self.terminate_session(oldest_session["session_id"])
            
            # Log security event
            self.log_security_event(
                "MAX_SESSIONS_EXCEEDED",
                oldest_session["session_id"],
                user_id,
                {"max_sessions": self.max_sessions_per_user}
            )

        # Generate session ID
        session_id = self.generate_session_id()
        
        # Create session record
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "username": username,
            "role": role,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "ip_address": ip_address,
            "device_info": device_info,
            "is_active": True,
            "security_level": "standard",
            "permissions": [],
            "activities": []
        }
        
        self.sessions[session_id] = session
        
        # Log security event
        self.log_security_event(
            "SESSION_CREATED",
            session_id,
            user_id,
            {
                "ip_address": ip_address,
                "device_info": device_info
            }
        )
        
        logger.info(f"✅ Session created: {session_id} for user: {username}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def update_session_activity(self, session_id: str, activity: Optional[str] = None) -> bool:
        """Update session last activity"""
        session = self.sessions.get(session_id)
        if not session or not session["is_active"]:
            return False

        # Check if session expired
        if self.is_session_expired(session):
            self.terminate_session(session_id)
            return False

        # Update activity
        session["last_activity"] = datetime.now()
        
        if activity:
            session["activities"].append({
                "activity": activity,
                "timestamp": datetime.now()
            })
            
            # Keep only last 50 activities
            if len(session["activities"]) > 50:
                session["activities"] = session["activities"][-50:]

        return True

    def is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Check if session is expired"""
        last_activity = session["last_activity"]
        return datetime.now() - last_activity > self.session_timeout

    def terminate_session(self, session_id: str) -> bool:
        """Terminate session"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        session["is_active"] = False
        session["terminated_at"] = datetime.now()
        
        # Log security event
        self.log_security_event(
            "SESSION_TERMINATED",
            session_id,
            session["user_id"],
            {"reason": "manual_termination"}
        )
        
        logger.info(f"✅ Session terminated: {session_id}")
        return True

    def terminate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Terminate all sessions for user"""
        terminated_count = 0
        
        for session_id, session in self.sessions.items():
            if (session["user_id"] == user_id and 
                session["is_active"] and 
                session_id != except_session):
                
                self.terminate_session(session_id)
                terminated_count += 1

        logger.info(f"✅ Terminated {terminated_count} sessions for user: {user_id}")
        return terminated_count

    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all sessions for user"""
        user_sessions = []
        
        for session in self.sessions.values():
            if session["user_id"] == user_id:
                if not active_only or session["is_active"]:
                    user_sessions.append(session)

        # Sort by last activity (most recent first)
        user_sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return user_sessions

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        cleaned_count = 0
        
        for session_id, session in list(self.sessions.items()):
            if session["is_active"] and self.is_session_expired(session):
                session["is_active"] = False
                session["terminated_at"] = datetime.now()
                
                # Log security event
                self.log_security_event(
                    "SESSION_EXPIRED",
                    session_id,
                    session["user_id"],
                    {"timeout": str(self.session_timeout)}
                )
                
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"✅ Cleaned up {cleaned_count} expired sessions")
        
        return cleaned_count

    def log_security_event(self, event_type: str, session_id: str, 
                          user_id: str, details: Dict[str, Any]):
        """Log security event"""
        event = {
            "event_type": event_type,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now(),
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

    def get_security_events(self, user_id: Optional[str] = None, 
                           event_type: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get security events"""
        events = self.security_events.copy()
        
        # Filter by user_id
        if user_id:
            events = [e for e in events if e["user_id"] == user_id]
        
        # Filter by event_type
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return events[:limit]

# Initialize session manager
session_manager = SessionManager()

@router.post("/create", response_model=SessionResponse)
async def create_session(request: SessionCreate, req: Request):
    """Create new session"""
    
    # Get client IP and user agent
    ip_address = request.ip_address or req.client.host
    device_info = request.device_info or req.headers.get("user-agent", "Unknown")
    
    session_id = session_manager.create_session(
        user_id=request.user_id,
        username=request.username,
        role=request.role,
        ip_address=ip_address,
        device_info=device_info
    )
    
    session = session_manager.get_session(session_id)
    
    return SessionResponse(
        session_id=session["session_id"],
        user_id=session["user_id"],
        username=session["username"],
        role=session["role"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        ip_address=session["ip_address"],
        device_info=session["device_info"],
        is_active=session["is_active"]
    )

@router.get("/list/{user_id}", response_model=SessionListResponse)
async def list_user_sessions(user_id: str, active_only: bool = True):
    """List sessions for user"""
    sessions = session_manager.get_user_sessions(user_id, active_only)
    
    session_responses = []
    for session in sessions:
        session_responses.append(SessionResponse(
            session_id=session["session_id"],
            user_id=session["user_id"],
            username=session["username"],
            role=session["role"],
            created_at=session["created_at"],
            last_activity=session["last_activity"],
            ip_address=session["ip_address"],
            device_info=session["device_info"],
            is_active=session["is_active"]
        ))
    
    active_sessions = [s for s in sessions if s["is_active"]]
    
    return SessionListResponse(
        sessions=session_responses,
        total_count=len(sessions),
        active_count=len(active_sessions)
    )

@router.post("/terminate/{session_id}")
async def terminate_session(session_id: str):
    """Terminate specific session"""
    if session_manager.terminate_session(session_id):
        return {"message": "Session terminated successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

@router.post("/terminate-user/{user_id}")
async def terminate_user_sessions(user_id: str, except_session: Optional[str] = None):
    """Terminate all sessions for user"""
    terminated_count = session_manager.terminate_user_sessions(user_id, except_session)
    
    return {
        "message": f"Terminated {terminated_count} sessions",
        "terminated_count": terminated_count
    }

@router.post("/activity/{session_id}")
async def update_session_activity(session_id: str, activity: Optional[str] = None):
    """Update session activity"""
    if session_manager.update_session_activity(session_id, activity):
        return {"message": "Session activity updated"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )

@router.get("/security-events")
async def get_security_events(user_id: Optional[str] = None, 
                             event_type: Optional[str] = None,
                             limit: int = 100):
    """Get security events"""
    events = session_manager.get_security_events(user_id, event_type, limit)
    
    return {
        "events": events,
        "total_count": len(events)
    }

@router.post("/cleanup")
async def cleanup_expired_sessions():
    """Clean up expired sessions"""
    cleaned_count = session_manager.cleanup_expired_sessions()
    
    return {
        "message": f"Cleaned up {cleaned_count} expired sessions",
        "cleaned_count": cleaned_count
    }

@router.get("/stats")
async def get_session_stats():
    """Get session statistics"""
    total_sessions = len(session_manager.sessions)
    active_sessions = sum(1 for s in session_manager.sessions.values() if s["is_active"])
    
    # Get unique users
    unique_users = len(set(s["user_id"] for s in session_manager.sessions.values() if s["is_active"]))
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "unique_active_users": unique_users,
        "session_timeout_hours": session_manager.session_timeout.total_seconds() / 3600,
        "max_sessions_per_user": session_manager.max_sessions_per_user
    }

# Export router
session_router = router