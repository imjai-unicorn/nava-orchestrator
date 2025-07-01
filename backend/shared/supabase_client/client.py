import os
from supabase import create_client, Client
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pathlib import Path

class SupabaseManager:
    """Supabase database manager"""
    
    def __init__(self):
        # Load .env from service directory
        service_env = Path(__file__).parent.parent.parent / "services" / "01-core" / "nava-logic-controller" / ".env"
        load_dotenv(service_env)
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            print(f"Looking for .env at: {service_env}")
            print(f"SUPABASE_URL: {url}")
            print(f"SUPABASE_SERVICE_ROLE_KEY: {key}")
            raise ValueError("Supabase credentials not found in environment")
        
        self.client: Client = create_client(url, key)
    
    # ... rest of methods เหมือนเดิม
    async def save_conversation(self, title: str = "New Conversation") -> str:
        try:
            result = self.client.table("conversations").insert({
                "title": title
            }).execute()
            return result.data[0]["id"]
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return None
    
    def test_connection(self):
        try:
            result = self.client.table("conversations").select("*").limit(1).execute()
            print(f"✅ Database connected! Found {len(result.data)} conversation(s)")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False