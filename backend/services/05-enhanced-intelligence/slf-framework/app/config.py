# backend/services/05-enhanced-intelligence/slf-framework/app/config.py
"""SLF Framework Configuration"""

import os
from pydantic import BaseSettings, Field

class SLFFrameworkConfig(BaseSettings):
    service_name: str = Field(default="slf-framework")
    port: int = Field(default=8010)
    enhancement_level: str = Field(default="standard")
    reasoning_depth: int = Field(default=3)
    max_prompt_length: int = Field(default=10000)
    
    class Config:
        env_file = ".env"

config = SLFFrameworkConfig()
