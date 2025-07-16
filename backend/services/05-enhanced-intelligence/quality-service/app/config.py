# backend/services/05-enhanced-intelligence/decision-engine/app/config.py
"""Decision Engine Configuration"""

import os
from pydantic import BaseSettings, Field
from typing import Dict, List

class DecisionEngineConfig(BaseSettings):
    service_name: str = Field(default="decision-engine")
    port: int = Field(default=8008)
    confidence_threshold: float = Field(default=0.8)
    max_alternatives: int = Field(default=3)
    decision_timeout: int = Field(default=10)
    
    class Config:
        env_file = ".env"

config = DecisionEngineConfig()

