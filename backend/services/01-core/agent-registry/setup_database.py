# setup_database.py
"""
Agent Registry Database Setup Script
Creates all necessary tables and initial data in Supabase
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Database imports
try:
    from supabase import create_client, Client
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError as e:
    print("❌ Missing database dependencies!")
    print("Run: pip install supabase psycopg2-binary")
    sys.exit(1)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not found, using system environment variables")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and schema creation"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.database_url = os.getenv("DATABASE_URL")
        
        self.supabase_client: Optional[Client] = None
        self.pg_connection = None
        
        # SQL schema file path
        self.schema_file = Path(__file__).parent / "schema" / "agent_registry_schema.sql"
    
    def validate_environment(self) -> bool:
        """Validate required environment variables"""
        logger.info("🔍 Validating environment configuration...")
        
        required_vars = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_service_key,
            "DATABASE_URL": self.database_url
        }
        
        missing_vars = []
        for var_name, var_value in required_vars.items():
            if not var_value:
                missing_vars.append(var_name)
                logger.error(f"❌ Missing: {var_name}")
            else:
                masked_value = var_value[:20] + "..." if len(var_value) > 20 else "***"
                logger.info(f"✅ {var_name}: {masked_value}")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            logger.info("💡 Please set these in your .env file:")
            for var in missing_vars:
                logger.info(f"   {var}=your-actual-value")
            return False
        
        logger.info("✅ Environment validation passed!")
        return True
    
    def connect_to_database(self) -> bool:
        """Establish database connections"""
        logger.info("🔗 Connecting to database...")
        
        try:
            # Supabase client
            self.supabase_client = create_client(
                self.supabase_url,
                self.supabase_service_key
            )
            
            # PostgreSQL connection for schema operations
            self.pg_connection = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor
            )
            
            # Test connections
            with self.pg_connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"✅ Connected to PostgreSQL: {version['version'][:50]}...")
            
            # Test Supabase
            response = self.supabase_client.table("schema_version").select("*").limit(1).execute()
            logger.info("✅ Supabase client connected successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def read_schema_file(self) -> str:
        """Read SQL schema from file or return inline schema"""
        try:
            if self.schema_file.exists():
                with open(self.schema_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning("⚠️ Schema file not found, using inline schema")
                return self.get_inline_schema()
        except Exception as e:
            logger.error(f"❌ Error reading schema file: {e}")
            return self.get_inline_schema()
    
    def get_inline_schema(self) -> str:
        """Return inline schema SQL (subset for basic setup)"""
        return """
        -- Enable UUID extension
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- AI Services Registry Table
        CREATE TABLE IF NOT EXISTS ai_services (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            service_id VARCHAR(50) UNIQUE NOT NULL,
            service_name VARCHAR(100) NOT NULL,
            service_url VARCHAR(255) NOT NULL,
            service_port INTEGER NOT NULL,
            service_type VARCHAR(20) NOT NULL CHECK (service_type IN ('external_ai', 'local_ai', 'hybrid')),
            models JSONB NOT NULL DEFAULT '[]',
            capabilities JSONB NOT NULL DEFAULT '[]',
            cost_per_1k_tokens DECIMAL(10,6) DEFAULT 0.0,
            max_tokens INTEGER DEFAULT 2048,
            timeout_seconds INTEGER DEFAULT 30,
            priority INTEGER DEFAULT 5,
            max_concurrent INTEGER DEFAULT 10,
            status VARCHAR(20) DEFAULT 'offline' CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'offline', 'maintenance')),
            last_health_check TIMESTAMPTZ,
            response_time_ms DECIMAL(10,2) DEFAULT 0.0,
            success_rate DECIMAL(5,4) DEFAULT 0.0,
            error_count INTEGER DEFAULT 0,
            total_requests INTEGER DEFAULT 0,
            current_load INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            is_active BOOLEAN DEFAULT true
        );
        
        -- Service Health History Table
        CREATE TABLE IF NOT EXISTS service_health_history (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            service_id UUID NOT NULL REFERENCES ai_services(id) ON DELETE CASCADE,
            check_timestamp TIMESTAMPTZ DEFAULT NOW(),
            status VARCHAR(20) NOT NULL,
            response_time_ms DECIMAL(10,2),
            error_message TEXT,
            health_score DECIMAL(3,2),
            metadata JSONB DEFAULT '{}'
        );
        
        -- Service Selection Logs Table
        CREATE TABLE IF NOT EXISTS service_selection_logs (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            request_id VARCHAR(100),
            user_id UUID,
            requested_capability VARCHAR(50),
            model_preference VARCHAR(50),
            cost_priority BOOLEAN DEFAULT false,
            performance_priority BOOLEAN DEFAULT false,
            selected_service_id UUID REFERENCES ai_services(id),
            selection_reason TEXT,
            confidence_score DECIMAL(3,2),
            failover_chain JSONB DEFAULT '[]',
            selection_time_ms DECIMAL(10,2),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        );
        
        -- Agent Registry Audit Logs Table
        CREATE TABLE IF NOT EXISTS agent_registry_audit_logs (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID,
            session_id VARCHAR(100),
            action VARCHAR(50) NOT NULL,
            resource_type VARCHAR(50) NOT NULL,
            resource_id VARCHAR(100),
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            success BOOLEAN DEFAULT true,
            error_message TEXT,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        );
        
        -- Registry System Config Table
        CREATE TABLE IF NOT EXISTS registry_system_config (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            config_category VARCHAR(50) NOT NULL,
            config_key VARCHAR(100) NOT NULL,
            config_value JSONB NOT NULL,
            description TEXT,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(config_category, config_key, is_active) WHERE is_active = true
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_ai_services_service_id ON ai_services(service_id);
        CREATE INDEX IF NOT EXISTS idx_ai_services_status ON ai_services(status);
        CREATE INDEX IF NOT EXISTS idx_health_history_service_id ON service_health_history(service_id);
        CREATE INDEX IF NOT EXISTS idx_selection_logs_service_id ON service_selection_logs(selected_service_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON agent_registry_audit_logs(timestamp);
        
        -- Schema version tracking
        CREATE TABLE IF NOT EXISTS schema_version (
            version VARCHAR(10) PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW(),
            description TEXT
        );
        """
    
    def execute_schema(self, schema_sql: str) -> bool:
        """Execute schema SQL"""
        logger.info("📋 Creating database schema...")
        
        try:
            with self.pg_connection.cursor() as cursor:
                # Split and execute statements
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            cursor.execute(statement)
                            logger.debug(f"✅ Executed statement {i+1}/{len(statements)}")
                        except Exception as e:
                            # Log warning but continue for non-critical errors
                            if "already exists" in str(e).lower():
                                logger.debug(f"⚠️ Statement {i+1}: {e}")
                            else:
                                logger.warning(f"⚠️ Statement {i+1} failed: {e}")
                
                # Commit all changes
                self.pg_connection.commit()
                logger.info("✅ Schema creation completed!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            if self.pg_connection:
                self.pg_connection.rollback()
            return False
    
    def insert_initial_data(self) -> bool:
        """Insert initial configuration data"""
        logger.info("📝 Inserting initial configuration data...")
        
        try:
            # Insert default system configurations
            default_configs = [
                ('health_monitoring', 'check_interval_seconds', 30, 'Health check interval in seconds'),
                ('health_monitoring', 'timeout_seconds', 10, 'Health check timeout in seconds'),
                ('health_monitoring', 'max_failures', 5, 'Maximum consecutive failures before marking unhealthy'),
                ('load_balancing', 'strategy', 'priority_based', 'Default load balancing strategy'),
                ('load_balancing', 'max_concurrent_per_service', 10, 'Maximum concurrent requests per service'),
                ('circuit_breaker', 'failure_threshold', 5, 'Circuit breaker failure threshold'),
                ('circuit_breaker', 'timeout_seconds', 60, 'Circuit breaker timeout in seconds'),
                ('alerting', 'response_time_threshold_ms', 3000, 'Alert threshold for response time'),
                ('caching', 'ttl_seconds', 300, 'Cache TTL in seconds'),
            ]
            
            for category, key, value, description in default_configs:
                try:
                    response = self.supabase_client.table("registry_system_config").upsert({
                        "config_category": category,
                        "config_key": key,
                        "config_value": value,
                        "description": description,
                        "is_active": True
                    }, on_conflict="config_category,config_key").execute()
                    
                    logger.debug(f"✅ Inserted config: {category}.{key}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert config {category}.{key}: {e}")
            
            # Insert schema version
            try:
                self.supabase_client.table("schema_version").upsert({
                    "version": "1.0.0",
                    "description": "Initial Agent Registry schema setup"
                }, on_conflict="version").execute()
                
                logger.info("✅ Schema version recorded!")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to record schema version: {e}")
            
            logger.info("✅ Initial data insertion completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initial data insertion failed: {e}")
            return False
    
    def register_default_services(self) -> bool:
        """Register default AI services"""
        logger.info("🤖 Registering default AI services...")
        
        default_services = [
            {
                "service_id": "gpt-client",
                "service_name": "OpenAI GPT",
                "service_url": "http://localhost:8002",
                "service_port": 8002,
                "service_type": "external_ai",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "capabilities": ["chat", "completion", "reasoning", "coding"],
                "cost_per_1k_tokens": 0.002,
                "max_tokens": 4096,
                "timeout_seconds": 30,
                "priority": 2
            },
            {
                "service_id": "claude-client",
                "service_name": "Anthropic Claude",
                "service_url": "http://localhost:8003",
                "service_port": 8003,
                "service_type": "external_ai",
                "models": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
                "capabilities": ["chat", "reasoning", "analysis", "writing"],
                "cost_per_1k_tokens": 0.0015,
                "max_tokens": 4096,
                "timeout_seconds": 45,
                "priority": 1
            },
            {
                "service_id": "gemini-client",
                "service_name": "Google Gemini",
                "service_url": "http://localhost:8004",
                "service_port": 8004,
                "service_type": "external_ai",
                "models": ["gemini-2.0-flash-exp", "gemini-2.5-pro"],
                "capabilities": ["chat", "multimodal", "search", "reasoning"],
                "cost_per_1k_tokens": 0.001,
                "max_tokens": 8192,
                "timeout_seconds": 25,
                "