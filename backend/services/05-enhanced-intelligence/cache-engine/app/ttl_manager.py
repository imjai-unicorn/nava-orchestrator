# backend/services/05-enhanced-intelligence/cache-engine/app/ttl_manager.py
"""
TTL Manager
Advanced Time-To-Live management for cache entries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
import asyncio
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Create router
ttl_router = APIRouter()

# Enums
class TTLStrategy(str, Enum):
    FIXED = "fixed"
    SLIDING = "sliding"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"

class ExpirationAction(str, Enum):
    DELETE = "delete"
    REFRESH = "refresh"
    ARCHIVE = "archive"
    NOTIFY = "notify"

# Models
class TTLRequest(BaseModel):
    key: str = Field(..., description="Cache key")
    ttl_seconds: int = Field(..., description="Time to live in seconds")
    strategy: TTLStrategy = Field(default=TTLStrategy.FIXED, description="TTL strategy")
    expiration_action: ExpirationAction = Field(default=ExpirationAction.DELETE, description="Action on expiration")
    priority: int = Field(default=1, description="Priority level (1-10)")
    auto_refresh: bool = Field(default=False, description="Auto-refresh on access")

class TTLUpdateRequest(BaseModel):
    key: str = Field(..., description="Cache key")
    new_ttl_seconds: Optional[int] = Field(None, description="New TTL in seconds")
    extend_by_seconds: Optional[int] = Field(None, description="Extend TTL by seconds")
    strategy: Optional[TTLStrategy] = Field(None, description="Update TTL strategy")

class TTLInfoResponse(BaseModel):
    key: str
    current_ttl_seconds: int
    expires_at: str
    strategy: str
    priority: int
    auto_refresh: bool
    access_count: int
    last_accessed: str
    expiration_action: str
    is_expired: bool

class TTLStatsResponse(BaseModel):
    total_entries: int
    expired_entries: int
    expiring_soon_count: int
    average_ttl_seconds: float
    strategy_distribution: Dict[str, int]
    expiration_actions_pending: Dict[str, int]

class TTLManager:
    """Advanced TTL Management System"""
    
    def __init__(self):
        # TTL storage
        self.ttl_entries = {}  # key -> TTL metadata
        
        # Strategy configurations
        self.strategy_config = {
            TTLStrategy.FIXED: {
                "description": "Fixed TTL that doesn't change",
                "refresh_factor": 1.0,
                "priority_modifier": 0.0
            },
            TTLStrategy.SLIDING: {
                "description": "TTL resets on access",
                "refresh_factor": 1.0,
                "priority_modifier": 0.0
            },
            TTLStrategy.ADAPTIVE: {
                "description": "TTL adapts based on usage patterns",
                "refresh_factor": 1.2,
                "priority_modifier": 0.1
            },
            TTLStrategy.PRIORITY_BASED: {
                "description": "TTL varies by priority level",
                "refresh_factor": 1.0,
                "priority_modifier": 0.2
            }
        }
        
        # Default TTL values by priority
        self.priority_ttl_map = {
            1: 300,    # 5 minutes - Low priority
            2: 600,    # 10 minutes
            3: 1800,   # 30 minutes
            4: 3600,   # 1 hour
            5: 7200,   # 2 hours - Medium priority
            6: 14400,  # 4 hours
            7: 28800,  # 8 hours
            8: 43200,  # 12 hours
            9: 86400,  # 24 hours
            10: 172800 # 48 hours - High priority
        }
        
        # Statistics
        self.stats = {
            "entries_created": 0,
            "entries_expired": 0,
            "entries_refreshed": 0,
            "cleanup_runs": 0,
            "expiration_actions": {
                "delete": 0,
                "refresh": 0,
                "archive": 0,
                "notify": 0
            }
        }
        
        # Start background cleanup
        self._start_cleanup_task()
    
    async def set_ttl(self, request: TTLRequest) -> Dict[str, Any]:
        """Set TTL for cache entry"""
        
        try:
            # Calculate actual TTL based on strategy
            actual_ttl = self._calculate_ttl(
                request.ttl_seconds,
                request.strategy,
                request.priority
            )
            
            # Calculate expiration time
            expires_at = datetime.now() + timedelta(seconds=actual_ttl)
            
            # Create TTL entry
            ttl_entry = {
                "key": request.key,
                "original_ttl": request.ttl_seconds,
                "current_ttl": actual_ttl,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "last_accessed": datetime.now(),
                "access_count": 0,
                "strategy": request.strategy,
                "priority": request.priority,
                "auto_refresh": request.auto_refresh,
                "expiration_action": request.expiration_action,
                "is_expired": False,
                "refresh_count": 0
            }
            
            # Store TTL entry
            self.ttl_entries[request.key] = ttl_entry
            self.stats["entries_created"] += 1
            
            logger.info(
                f"⏰ TTL set: {request.key} → {actual_ttl}s "
                f"({request.strategy}, priority: {request.priority})"
            )
            
            return {
                "success": True,
                "key": request.key,
                "ttl_seconds": actual_ttl,
                "expires_at": expires_at.isoformat(),
                "strategy": request.strategy.value
            }
            
        except Exception as e:
            logger.error(f"❌ Set TTL error for {request.key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_ttl_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get TTL information for cache entry"""
        
        try:
            if key not in self.ttl_entries:
                return None
            
            entry = self.ttl_entries[key]
            
            # Check if expired
            is_expired = self._is_expired(entry)
            if is_expired and not entry.get("is_expired", False):
                entry["is_expired"] = True
                await self._handle_expiration(key, entry)
            
            # Calculate remaining TTL
            now = datetime.now()
            expires_at = entry["expires_at"]
            
            if isinstance(expires_at, datetime):
                remaining_seconds = max(0, int((expires_at - now).total_seconds()))
            else:
                remaining_seconds = 0
            
            return {
                "key": key,
                "current_ttl_seconds": remaining_seconds,
                "expires_at": entry["expires_at"].isoformat() if isinstance(entry["expires_at"], datetime) else str(entry["expires_at"]),
                "strategy": entry["strategy"].value if hasattr(entry["strategy"], 'value') else str(entry["strategy"]),
                "priority": entry["priority"],
                "auto_refresh": entry["auto_refresh"],
                "access_count": entry["access_count"],
                "last_accessed": entry["last_accessed"].isoformat() if isinstance(entry["last_accessed"], datetime) else str(entry["last_accessed"]),
                "expiration_action": entry["expiration_action"].value if hasattr(entry["expiration_action"], 'value') else str(entry["expiration_action"]),
                "is_expired": is_expired
            }
            
        except Exception as e:
            logger.error(f"❌ Get TTL info error for {key}: {e}")
            return None
    
    async def update_ttl(self, request: TTLUpdateRequest) -> Dict[str, Any]:
        """Update TTL for existing cache entry"""
        
        try:
            if request.key not in self.ttl_entries:
                return {"success": False, "error": "Key not found"}
            
            entry = self.ttl_entries[request.key]
            
            # Calculate new TTL
            if request.new_ttl_seconds is not None:
                new_ttl = request.new_ttl_seconds
            elif request.extend_by_seconds is not None:
                current_remaining = max(0, int((entry["expires_at"] - datetime.now()).total_seconds()))
                new_ttl = current_remaining + request.extend_by_seconds
            else:
                return {"success": False, "error": "No TTL update specified"}
            
            # Update strategy if provided
            if request.strategy:
                entry["strategy"] = request.strategy
                new_ttl = self._calculate_ttl(new_ttl, request.strategy, entry["priority"])
            
            # Update expiration time
            entry["current_ttl"] = new_ttl
            entry["expires_at"] = datetime.now() + timedelta(seconds=new_ttl)
            entry["is_expired"] = False
            
            logger.info(f"⏰ TTL updated: {request.key} → {new_ttl}s")
            
            return {
                "success": True,
                "key": request.key,
                "new_ttl_seconds": new_ttl,
                "expires_at": entry["expires_at"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Update TTL error for {request.key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def access_entry(self, key: str) -> Dict[str, Any]:
        """Record access to cache entry (for sliding/adaptive TTL)"""
        
        try:
            if key not in self.ttl_entries:
                return {"success": False, "error": "Key not found"}
            
            entry = self.ttl_entries[key]
            
            # Check if expired
            if self._is_expired(entry):
                return {"success": False, "error": "Entry expired"}
            
            # Update access info
            entry["last_accessed"] = datetime.now()
            entry["access_count"] += 1
            
            # Handle strategy-specific logic
            if entry["strategy"] == TTLStrategy.SLIDING:
                # Reset TTL on access
                new_ttl = entry["original_ttl"]
                entry["expires_at"] = datetime.now() + timedelta(seconds=new_ttl)
                entry["refresh_count"] += 1
                self.stats["entries_refreshed"] += 1
                
            elif entry["strategy"] == TTLStrategy.ADAPTIVE:
                # Extend TTL based on access frequency
                access_frequency = entry["access_count"] / max(1, (datetime.now() - entry["created_at"]).total_seconds() / 3600)
                
                if access_frequency > 1.0:  # More than 1 access per hour
                    extension = min(3600, entry["original_ttl"] * 0.5)  # Extend up to 1 hour
                    entry["expires_at"] += timedelta(seconds=extension)
                    entry["refresh_count"] += 1
                    self.stats["entries_refreshed"] += 1
            
            # Auto-refresh if enabled
            if entry.get("auto_refresh", False) and entry["access_count"] % 10 == 0:
                await self._auto_refresh_entry(key, entry)
            
            return {
                "success": True,
                "key": key,
                "access_count": entry["access_count"],
                "ttl_remaining": max(0, int((entry["expires_at"] - datetime.now()).total_seconds()))
            }
            
        except Exception as e:
            logger.error(f"❌ Access entry error for {key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def remove_ttl(self, key: str) -> Dict[str, Any]:
        """Remove TTL entry"""
        
        try:
            if key in self.ttl_entries:
                del self.ttl_entries[key]
                
                logger.info(f"🗑️ TTL removed: {key}")
                
                return {"success": True, "key": key}
            else:
                return {"success": False, "error": "Key not found"}
                
        except Exception as e:
            logger.error(f"❌ Remove TTL error for {key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_expiring_soon(self, within_seconds: int = 300) -> List[Dict[str, Any]]:
        """Get entries expiring within specified time"""
        
        try:
            expiring_entries = []
            cutoff_time = datetime.now() + timedelta(seconds=within_seconds)
            
            for key, entry in self.ttl_entries.items():
                if not entry.get("is_expired", False) and entry["expires_at"] <= cutoff_time:
                    remaining_seconds = max(0, int((entry["expires_at"] - datetime.now()).total_seconds()))
                    
                    expiring_entries.append({
                        "key": key,
                        "expires_in_seconds": remaining_seconds,
                        "expires_at": entry["expires_at"].isoformat(),
                        "priority": entry["priority"],
                        "strategy": entry["strategy"].value if hasattr(entry["strategy"], 'value') else str(entry["strategy"]),
                        "expiration_action": entry["expiration_action"].value if hasattr(entry["expiration_action"], 'value') else str(entry["expiration_action"])
                    })
            
            # Sort by expiration time
            expiring_entries.sort(key=lambda x: x["expires_in_seconds"])
            
            return expiring_entries
            
        except Exception as e:
            logger.error(f"❌ Get expiring soon error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get TTL manager statistics"""
        
        try:
            total_entries = len(self.ttl_entries)
            expired_count = sum(1 for entry in self.ttl_entries.values() if self._is_expired(entry))
            expiring_soon_count = len(await self.get_expiring_soon(300))  # Next 5 minutes
            
            # Calculate average TTL
            if total_entries > 0:
                total_ttl = sum(
                    max(0, int((entry["expires_at"] - datetime.now()).total_seconds()))
                    for entry in self.ttl_entries.values()
                    if not self._is_expired(entry)
                )
                avg_ttl = total_ttl / max(1, total_entries - expired_count)
            else:
                avg_ttl = 0.0
            
            # Strategy distribution
            strategy_dist = {}
            for entry in self.ttl_entries.values():
                strategy = entry["strategy"].value if hasattr(entry["strategy"], 'value') else str(entry["strategy"])
                strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
            
            # Expiration actions pending
            expiration_actions = {}
            for entry in self.ttl_entries.values():
                if self._is_expired(entry):
                    action = entry["expiration_action"].value if hasattr(entry["expiration_action"], 'value') else str(entry["expiration_action"])
                    expiration_actions[action] = expiration_actions.get(action, 0) + 1
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "expiring_soon_count": expiring_soon_count,
                "average_ttl_seconds": round(avg_ttl, 2),
                "strategy_distribution": strategy_dist,
                "expiration_actions_pending": expiration_actions,
                "lifetime_stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Get TTL stats error: {e}")
            return {"error": str(e)}
    
    def _calculate_ttl(self, base_ttl: int, strategy: TTLStrategy, priority: int) -> int:
        """Calculate actual TTL based on strategy and priority"""
        
        if strategy == TTLStrategy.PRIORITY_BASED:
            # Use priority-based TTL
            priority_ttl = self.priority_ttl_map.get(priority, base_ttl)
            return max(base_ttl, priority_ttl)
        
        elif strategy == TTLStrategy.ADAPTIVE:
            # Slightly longer initial TTL for adaptive strategy
            return int(base_ttl * 1.2)
        
        else:
            # Fixed or sliding - use base TTL
            return base_ttl
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if TTL entry is expired"""
        
        try:
            expires_at = entry.get("expires_at")
            if expires_at and isinstance(expires_at, datetime):
                return datetime.now() > expires_at
            return False
            
        except Exception as e:
            logger.error(f"❌ Check expiration error: {e}")
            return True  # Assume expired on error
    
    async def _handle_expiration(self, key: str, entry: Dict[str, Any]):
        """Handle expired entry based on expiration action"""
        
        try:
            action = entry.get("expiration_action", ExpirationAction.DELETE)
            
            if action == ExpirationAction.DELETE:
                # Mark for deletion
                self.stats["expiration_actions"]["delete"] += 1
                logger.debug(f"⏰ Expired for deletion: {key}")
                
            elif action == ExpirationAction.REFRESH:
                # Auto-refresh entry
                await self._auto_refresh_entry(key, entry)
                self.stats["expiration_actions"]["refresh"] += 1
                logger.debug(f"⏰ Auto-refreshed: {key}")
                
            elif action == ExpirationAction.ARCHIVE:
                # Mark for archival
                entry["archived"] = True
                self.stats["expiration_actions"]["archive"] += 1
                logger.debug(f"⏰ Archived: {key}")
                
            elif action == ExpirationAction.NOTIFY:
                # Log notification
                self.stats["expiration_actions"]["notify"] += 1
                logger.info(f"⏰ Expiration notification: {key}")
            
            self.stats["entries_expired"] += 1
            
        except Exception as e:
            logger.error(f"❌ Handle expiration error for {key}: {e}")
    
    async def _auto_refresh_entry(self, key: str, entry: Dict[str, Any]):
        """Auto-refresh cache entry"""
        
        try:
            # Extend TTL based on access pattern
            access_frequency = entry.get("access_count", 0) / max(1, entry.get("refresh_count", 0) + 1)
            
            if access_frequency > 5:  # High access frequency
                extension = entry["original_ttl"]
            elif access_frequency > 2:  # Medium access frequency
                extension = entry["original_ttl"] // 2
            else:  # Low access frequency
                extension = entry["original_ttl"] // 4
            
            # Update expiration time
            entry["expires_at"] = datetime.now() + timedelta(seconds=extension)
            entry["refresh_count"] = entry.get("refresh_count", 0) + 1
            entry["is_expired"] = False
            
            logger.debug(f"🔄 Auto-refreshed {key} for {extension}s")
            
        except Exception as e:
            logger.error(f"❌ Auto-refresh error for {key}: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        
        async def cleanup_expired():
            while True:
                try:
                    await self._cleanup_expired_entries()
                    await asyncio.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"❌ TTL cleanup task error: {e}")
                    await asyncio.sleep(30)  # Retry after 30 seconds
        
        # Start cleanup task
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(cleanup_expired())
        except:
            # If no event loop, skip background task
            pass
    
    async def _cleanup_expired_entries(self):
        """Cleanup expired TTL entries"""
        
        try:
            expired_keys = []
            
            for key, entry in self.ttl_entries.items():
                if self._is_expired(entry) and not entry.get("is_expired", False):
                    await self._handle_expiration(key, entry)
                    
                    # Remove entries marked for deletion
                    if entry.get("expiration_action") == ExpirationAction.DELETE:
                        expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self.ttl_entries[key]
            
            if expired_keys:
                self.stats["cleanup_runs"] += 1
                logger.info(f"🧹 TTL cleanup: removed {len(expired_keys)} expired entries")
                
        except Exception as e:
            logger.error(f"❌ TTL cleanup error: {e}")

# Initialize TTL manager
ttl_manager = TTLManager()

@ttl_router.post("/set")
async def set_ttl(request: TTLRequest):
    """Set TTL for cache entry"""
    
    try:
        result = await ttl_manager.set_ttl(request)
        
        if result.get("success"):
            logger.info(f"⏰ TTL set: {request.key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Set TTL endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.get("/info/{key}", response_model=TTLInfoResponse)
async def get_ttl_info(key: str):
    """Get TTL information for cache entry"""
    
    try:
        info = await ttl_manager.get_ttl_info(key)
        
        if info is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return TTLInfoResponse(**info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get TTL info endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.put("/update")
async def update_ttl(request: TTLUpdateRequest):
    """Update TTL for existing cache entry"""
    
    try:
        result = await ttl_manager.update_ttl(request)
        
        if result.get("success"):
            logger.info(f"⏰ TTL updated: {request.key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Update TTL endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.post("/access/{key}")
async def access_entry(key: str):
    """Record access to cache entry"""
    
    try:
        result = await ttl_manager.access_entry(key)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Access entry endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.delete("/remove/{key}")
async def remove_ttl(key: str):
    """Remove TTL entry"""
    
    try:
        result = await ttl_manager.remove_ttl(key)
        
        if result.get("success"):
            logger.info(f"🗑️ TTL removed: {key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Remove TTL endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.get("/expiring-soon")
async def get_expiring_soon(within_seconds: int = 300):
    """Get entries expiring within specified time"""
    
    try:
        expiring = await ttl_manager.get_expiring_soon(within_seconds)
        
        return {
            "expiring_entries": expiring,
            "count": len(expiring),
            "within_seconds": within_seconds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Get expiring soon endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.get("/stats", response_model=TTLStatsResponse)
async def get_ttl_stats():
    """Get TTL manager statistics"""
    
    try:
        stats = await ttl_manager.get_stats()
        
        return TTLStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ Get TTL stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ttl_router.get("/strategies")
async def get_ttl_strategies():
    """Get available TTL strategies"""
    
    return {
        "strategies": {
            strategy.value: config for strategy, config in ttl_manager.strategy_config.items()
        },
        "expiration_actions": [action.value for action in ExpirationAction],
        "priority_ttl_map": ttl_manager.priority_ttl_map,
        "default_strategy": TTLStrategy.FIXED.value,
        "timestamp": datetime.now().isoformat()
    }

@ttl_router.post("/cleanup")
async def manual_cleanup():
    """Manually trigger TTL cleanup"""
    
    try:
        await ttl_manager._cleanup_expired_entries()
        
        return {
            "success": True,
            "message": "Manual cleanup completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual cleanup endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))