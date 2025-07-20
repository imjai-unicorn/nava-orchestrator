import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # Service
    SERVICE_NAME = os.getenv("SERVICE_NAME", "nava-logic-controller")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8005))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()