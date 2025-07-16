# backend/services/05-enhanced-intelligence/cache-engine/app/config.py
"""Cache Engine Configuration"""

import os
from pydantic import BaseSettings, Field

class CacheEngineConfig(BaseSettings):
    service_name: str = Field(default="cache-engine")
    port: int = Field(default=8013)
    memory_max_size: int = Field(default=1000)
    default_ttl: int = Field(default=3600)
    similarity_threshold: float = Field(default=0.8)
    
    class Config:
        env_file = ".env"

config = CacheEngineConfig()