import os
from supabase import create_client, Client
from typing import Optional

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