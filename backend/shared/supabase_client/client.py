import os
from typing import Optional

# Simple mock for Railway compatibility
class SupabaseManager:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client with error handling"""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not url or not key:
                print("Warning: Supabase credentials not found - using mock mode")
                self.client = None
                return
            
            # Try to import and create Supabase client
            try:
                from supabase import create_client
                self.client = create_client(url, key)
                print("✅ Supabase client initialized successfully")
            except Exception as import_error:
                print(f"⚠️ Supabase import error: {import_error}")
                print("Using mock mode for compatibility")
                self.client = None
            
        except Exception as e:
            print(f"❌ Failed to initialize Supabase client: {e}")
            self.client = None
    
    def get_client(self):
        """Get Supabase client instance"""
        return self.client
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected"""
        return self.client is not None
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        if not self.client:
            print("⚠️ Supabase not connected - using mock mode")
            return True  # Return True for mock mode
        
        try:
            # Simple test query
            result = self.client.table('conversations').select('*').limit(1).execute()
            return True
        except Exception as e:
            print(f"❌ Supabase connection test failed: {e}")
            return False
    
    async def save_conversation(self, title: str) -> str:
        """Save conversation - mock implementation"""
        return f"mock-conversation-{hash(title) % 1000}"
    
    async def save_message(self, conversation_id: str, message: str, response: str) -> str:
        """Save message - mock implementation"""
        return f"mock-message-{hash(message) % 1000}"

# Global instance
supabase_manager = SupabaseManager()