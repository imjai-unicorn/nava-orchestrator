import os
from supabase import create_client, Client
from typing import Optional
from datetime import datetime
import json

class SupabaseManager:
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client with error handling"""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not url or not key:
                print("Warning: Supabase credentials not found")
                return None
            
            # Simple client creation without proxy
            self.client = create_client(url, key)
            print("✅ Supabase client initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Supabase client: {e}")
            self.client = None
    
    def get_client(self) -> Optional[Client]:
        """Get Supabase client instance"""
        return self.client
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected"""
        return self.client is not None
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        if not self.client:
            return False
        
        try:
            # Simple test query
            result = self.client.table('conversations').select('*').limit(1).execute()
            return True
        except Exception as e:
            print(f"❌ Supabase connection test failed: {e}")
            return False


# Global instance
supabase_manager = SupabaseManager()

async def save_conversation(self, conversation_data: dict) -> dict:
    """Save conversation to Supabase - เพิ่ม method ที่หายไป"""
    if not self.client:
        return {"success": False, "error": "No Supabase connection"}
    
    try:
        # Prepare conversation data
        data = {
            "conversation_id": conversation_data.get("conversation_id"),
            "user_id": conversation_data.get("user_id"), 
            "user_message": conversation_data.get("user_input"),
            "ai_response": conversation_data.get("ai_response"),
            "selected_model": conversation_data.get("selected_model"),
            "confidence": conversation_data.get("confidence"),
            "reasoning": json.dumps(conversation_data.get("reasoning", {})),
            "behavior_pattern": conversation_data.get("behavior_pattern"),
            "task_type": conversation_data.get("task_type"),
            "complexity_score": conversation_data.get("complexity_score"),
            "processing_time_ms": conversation_data.get("processing_time_ms"),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert to conversations table
        result = self.client.table("conversations").insert(data).execute()
        
        return {
            "success": True,
            "conversation_id": conversation_data.get("conversation_id"),
            "record_id": result.data[0]["id"] if result.data else None
        }
        
    except Exception as e:
        print(f"❌ Error saving conversation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "conversation_id": conversation_data.get("conversation_id")
        }

async def get_conversation_history(self, user_id: str, limit: int = 10):
    """Get conversation history for user"""
    if not self.client:
        return []
    
    try:
        result = self.client.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data
        
    except Exception as e:
        print(f"❌ Error getting conversation history: {str(e)}")
        return []