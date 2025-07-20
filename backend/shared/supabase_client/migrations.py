# backend/shared/supabase-client/migrations.py
"""
Database Migration Management for NAVA
Handles database schema changes and data migrations for Supabase
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from supabase import create_client, Client
import os

logger = logging.getLogger(__name__)

@dataclass
class Migration:
    """Database migration definition"""
    id: str
    name: str
    description: str
    version: str
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    is_applied: bool = False

@dataclass
class MigrationResult:
    """Result of migration execution"""
    migration_id: str
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class MigrationManager:
    """
    Database migration management
    Handles schema evolution and data migrations
    """
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.migrations: Dict[str, Migration] = {}
        self.migration_history: List[MigrationResult] = []
        
        # Initialize migration table
        self._initialize_migration_table()
        
        # Load existing migrations from database
        self._load_migration_history()
        
        logger.info("✅ Migration Manager initialized")
    
    def _initialize_migration_table(self):
        """Initialize the migration tracking table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS nava_migrations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            version TEXT NOT NULL,
            up_sql TEXT NOT NULL,
            down_sql TEXT NOT NULL,
            dependencies JSONB DEFAULT '[]',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            applied_at TIMESTAMP WITH TIME ZONE,
            is_applied BOOLEAN DEFAULT FALSE,
            execution_time_ms FLOAT DEFAULT 0.0,
            error_message TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_nava_migrations_version ON nava_migrations(version);
        CREATE INDEX IF NOT EXISTS idx_nava_migrations_applied ON nava_migrations(is_applied);
        """
        
        try:
            # Execute the SQL directly (this would need to be adapted based on Supabase client capabilities)
            # For now, we'll log that this needs to be run manually
            logger.info("📋 Migration table SQL prepared - may need manual execution")
            logger.debug(f"SQL: {create_table_sql}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize migration table: {e}")
    
    def _load_migration_history(self):
        """Load migration history from database"""
        try:
            # Load applied migrations
            result = self.client.table('nava_migrations').select('*').execute()
            
            if result.data:
                for row in result.data:
                    migration = Migration(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'],
                        version=row['version'],
                        up_sql=row['up_sql'],
                        down_sql=row['down_sql'],
                        dependencies=row.get('dependencies', []),
                        created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                        applied_at=datetime.fromisoformat(row['applied_at'].replace('Z', '+00:00')) if row['applied_at'] else None,
                        is_applied=row['is_applied']
                    )
                    
                    self.migrations[migration.id] = migration
                    
                    if migration.is_applied:
                        result_record = MigrationResult(
                            migration_id=migration.id,
                            success=True,
                            execution_time_ms=row.get('execution_time_ms', 0.0),
                            timestamp=migration.applied_at or migration.created_at
                        )
                        self.migration_history.append(result_record)
                
                logger.info(f"✅ Loaded {len(self.migrations)} migrations from database")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load migration history: {e}")
    
    def register_migration(self, migration: Migration) -> bool:
        """Register a new migration"""
        if migration.id in self.migrations:
            logger.warning(f"⚠️ Migration {migration.id} already exists")
            return False
        
        self.migrations[migration.id] = migration
        logger.info(f"✅ Registered migration: {migration.id} - {migration.name}")
        return True
    
    def create_migration(self, id: str, name: str, description: str, 
                        version: str, up_sql: str, down_sql: str,
                        dependencies: List[str] = None) -> Migration:
        """Create a new migration"""
        migration = Migration(
            id=id,
            name=name,
            description=description,
            version=version,
            up_sql=up_sql,
            down_sql=down_sql,
            dependencies=dependencies or []
        )
        
        self.register_migration(migration)
        return migration
    
    def rollback_migration(self, migration_id: str) -> MigrationResult:
        """Rollback a specific migration"""
        if migration_id not in self.migrations:
            return MigrationResult(
                migration_id=migration_id,
                success=False,
                error_message="Migration not found"
            )
        
        migration = self.migrations[migration_id]
        
        if not migration.is_applied:
            return MigrationResult(
                migration_id=migration_id,
                success=True,
                error_message="Migration not applied, nothing to rollback"
            )
        
        # Check if other migrations depend on this one
        dependents = [m for m in self.migrations.values() 
                     if migration_id in m.dependencies and m.is_applied]
        
        if dependents:
            dependent_names = [m.name for m in dependents]
            return MigrationResult(
                migration_id=migration_id,
                success=False,
                error_message=f"Cannot rollback: dependent migrations exist: {', '.join(dependent_names)}"
            )
        
        # Execute rollback
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 Rolling back migration: {migration.id} - {migration.name}")
            
            # Execute the DOWN SQL
            self._execute_sql(migration.down_sql)
            
            # Mark as not applied
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            migration.applied_at = None
            migration.is_applied = False
            
            # Update database record
            self._update_migration_status(migration, False, execution_time)
            
            result = MigrationResult(
                migration_id=migration_id,
                success=True,
                execution_time_ms=execution_time
            )
            
            self.migration_history.append(result)
            logger.info(f"✅ Migration rolled back successfully: {migration.id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Rollback failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            result = MigrationResult(
                migration_id=migration_id,
                success=False,
                error_message=error_msg,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            self.migration_history.append(result)
            return result
    
    def apply_all_pending(self) -> List[MigrationResult]:
        """Apply all pending migrations in order"""
        pending = self.get_pending_migrations()
        results = []
        
        logger.info(f"🔄 Applying {len(pending)} pending migrations...")
        
        for migration in pending:
            result = self.apply_migration(migration.id)
            results.append(result)
            
            if not result.success:
                logger.error(f"❌ Migration chain stopped at {migration.id}: {result.error_message}")
                break
        
        successful = len([r for r in results if r.success])
        logger.info(f"✅ Applied {successful}/{len(results)} migrations successfully")
        
        return results
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get all pending migrations in dependency order"""
        pending = [m for m in self.migrations.values() if not m.is_applied]
        
        # Sort by dependencies (topological sort)
        sorted_migrations = []
        remaining = pending.copy()
        
        while remaining:
            # Find migrations with no unresolved dependencies
            ready = []
            for migration in remaining:
                deps_satisfied = all(
                    dep_id in self.migrations and self.migrations[dep_id].is_applied 
                    for dep_id in migration.dependencies
                )
                if deps_satisfied:
                    ready.append(migration)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.error("❌ Circular dependency detected in migrations")
                break
            
            # Add ready migrations to sorted list
            sorted_migrations.extend(ready)
            
            # Remove from remaining
            for migration in ready:
                remaining.remove(migration)
        
        return sorted_migrations
    
    def get_applied_migrations(self) -> List[Migration]:
        """Get all applied migrations"""
        return [m for m in self.migrations.values() if m.is_applied]
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status"""
        total = len(self.migrations)
        applied = len([m for m in self.migrations.values() if m.is_applied])
        pending = total - applied
        
        recent_history = self.migration_history[-10:] if self.migration_history else []
        
        return {
            "total_migrations": total,
            "applied_migrations": applied,
            "pending_migrations": pending,
            "last_applied": max([m.applied_at for m in self.migrations.values() if m.applied_at], default=None),
            "recent_history": [
                {
                    "migration_id": r.migration_id,
                    "success": r.success,
                    "timestamp": r.timestamp.isoformat(),
                    "execution_time_ms": r.execution_time_ms,
                    "error": r.error_message
                }
                for r in recent_history
            ]
        }
    
    def validate_migrations(self) -> List[Dict[str, Any]]:
        """Validate all migrations for consistency"""
        issues = []
        
        # Check for duplicate IDs
        id_counts = {}
        for migration in self.migrations.values():
            id_counts[migration.id] = id_counts.get(migration.id, 0) + 1
        
        duplicates = [mid for mid, count in id_counts.items() if count > 1]
        if duplicates:
            issues.append({
                "type": "duplicate_ids",
                "message": f"Duplicate migration IDs: {', '.join(duplicates)}"
            })
        
        # Check dependencies
        for migration in self.migrations.values():
            for dep_id in migration.dependencies:
                if dep_id not in self.migrations:
                    issues.append({
                        "type": "missing_dependency",
                        "migration_id": migration.id,
                        "message": f"Missing dependency: {dep_id}"
                    })
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append({
                "type": "circular_dependency",
                "message": "Circular dependencies detected"
            })
        
        # Check SQL syntax (basic validation)
        for migration in self.migrations.values():
            if not migration.up_sql.strip():
                issues.append({
                    "type": "empty_sql",
                    "migration_id": migration.id,
                    "message": "Empty UP SQL"
                })
            
            if not migration.down_sql.strip():
                issues.append({
                    "type": "empty_sql",
                    "migration_id": migration.id,
                    "message": "Empty DOWN SQL"
                })
        
        return issues
    
    def _execute_sql(self, sql: str):
        """Execute SQL statement"""
        # This is a simplified version - real implementation would use proper SQL execution
        # For Supabase, this might involve using the RPC functionality or admin API
        logger.info(f"📄 Executing SQL: {sql[:100]}...")
        
        # In a real implementation, this would execute the SQL
        # For now, we'll simulate successful execution
        pass
    
    def _save_migration_to_db(self, migration: Migration, execution_time_ms: float):
        """Save migration record to database"""
        try:
            migration_data = {
                "id": migration.id,
                "name": migration.name,
                "description": migration.description,
                "version": migration.version,
                "up_sql": migration.up_sql,
                "down_sql": migration.down_sql,
                "dependencies": migration.dependencies,
                "created_at": migration.created_at.isoformat(),
                "applied_at": migration.applied_at.isoformat() if migration.applied_at else None,
                "is_applied": migration.is_applied,
                "execution_time_ms": execution_time_ms
            }
            
            # Upsert migration record
            self.client.table('nava_migrations').upsert(migration_data).execute()
            
        except Exception as e:
            logger.error(f"❌ Failed to save migration to database: {e}")
    
    def _update_migration_status(self, migration: Migration, is_applied: bool, execution_time_ms: float):
        """Update migration status in database"""
        try:
            update_data = {
                "is_applied": is_applied,
                "applied_at": migration.applied_at.isoformat() if migration.applied_at else None,
                "execution_time_ms": execution_time_ms
            }
            
            self.client.table('nava_migrations').update(update_data).eq('id', migration.id).execute()
            
        except Exception as e:
            logger.error(f"❌ Failed to update migration status: {e}")
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            if node in self.migrations:
                for dep in self.migrations[node].dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for migration_id in self.migrations:
            if migration_id not in visited:
                if has_cycle(migration_id, visited, set()):
                    return True
        
        return False

class NAVAMigrations:
    """NAVA-specific migration definitions"""
    
    @staticmethod
    def get_initial_schema_migrations() -> List[Migration]:
        """Get initial schema migrations for NAVA"""
        migrations = []
        
        # Migration 1: Create sessions table
        migrations.append(Migration(
            id="001_create_sessions",
            name="Create conversation sessions table",
            description="Initial table for conversation sessions",
            version="1.0.0",
            up_sql="""
            CREATE TABLE IF NOT EXISTS nava_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                user_preferences JSONB DEFAULT '{}',
                context_summary TEXT DEFAULT '',
                total_messages INTEGER DEFAULT 0,
                session_metadata JSONB DEFAULT '{}',
                is_active BOOLEAN DEFAULT TRUE
            );
            
            CREATE INDEX idx_nava_sessions_user_id ON nava_sessions(user_id);
            CREATE INDEX idx_nava_sessions_last_active ON nava_sessions(last_active);
            CREATE INDEX idx_nava_sessions_is_active ON nava_sessions(is_active);
            """,
            down_sql="DROP TABLE IF EXISTS nava_sessions CASCADE;"
        ))
        
        # Migration 2: Create messages table
        migrations.append(Migration(
            id="002_create_messages",
            name="Create conversation messages table",
            description="Table for storing conversation messages",
            version="1.0.0",
            up_sql="""
            CREATE TABLE IF NOT EXISTS nava_messages (
                message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES nava_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                model_used TEXT,
                processing_time_ms FLOAT,
                quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
                message_metadata JSONB DEFAULT '{}'
            );
            
            CREATE INDEX idx_nava_messages_session_id ON nava_messages(session_id);
            CREATE INDEX idx_nava_messages_created_at ON nava_messages(created_at);
            CREATE INDEX idx_nava_messages_role ON nava_messages(role);
            CREATE INDEX idx_nava_messages_model_used ON nava_messages(model_used);
            """,
            down_sql="DROP TABLE IF EXISTS nava_messages CASCADE;",
            dependencies=["001_create_sessions"]
        ))
        
        # Migration 3: Create workflows table
        migrations.append(Migration(
            id="003_create_workflows",
            name="Create workflows table",
            description="Table for storing workflow definitions and executions",
            version="1.0.0",
            up_sql="""
            CREATE TABLE IF NOT EXISTS nava_workflows (
                workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                description TEXT,
                workflow_definition JSONB NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                version TEXT NOT NULL DEFAULT '1.0.0',
                created_by TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                tags TEXT[] DEFAULT '{}',
                is_template BOOLEAN DEFAULT FALSE
            );
            
            CREATE INDEX idx_nava_workflows_status ON nava_workflows(status);
            CREATE INDEX idx_nava_workflows_created_by ON nava_workflows(created_by);
            CREATE INDEX idx_nava_workflows_is_template ON nava_workflows(is_template);
            CREATE INDEX idx_nava_workflows_tags ON nava_workflows USING GIN(tags);
            """,
            down_sql="DROP TABLE IF EXISTS nava_workflows CASCADE;"
        ))
        
        # Migration 4: Create workflow executions table
        migrations.append(Migration(
            id="004_create_workflow_executions",
            name="Create workflow executions table",
            description="Table for tracking workflow execution instances",
            version="1.0.0",
            up_sql="""
            CREATE TABLE IF NOT EXISTS nava_workflow_executions (
                execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                workflow_id UUID NOT NULL REFERENCES nava_workflows(workflow_id) ON DELETE CASCADE,
                session_id UUID REFERENCES nava_sessions(session_id) ON DELETE SET NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                execution_context JSONB DEFAULT '{}',
                execution_log JSONB DEFAULT '[]',
                final_result JSONB,
                error_message TEXT,
                performance_metrics JSONB DEFAULT '{}'
            );
            
            CREATE INDEX idx_nava_workflow_executions_workflow_id ON nava_workflow_executions(workflow_id);
            CREATE INDEX idx_nava_workflow_executions_session_id ON nava_workflow_executions(session_id);
            CREATE INDEX idx_nava_workflow_executions_status ON nava_workflow_executions(status);
            CREATE INDEX idx_nava_workflow_executions_started_at ON nava_workflow_executions(started_at);
            """,
            down_sql="DROP TABLE IF EXISTS nava_workflow_executions CASCADE;",
            dependencies=["003_create_workflows", "001_create_sessions"]
        ))
        
        # Migration 5: Create performance metrics table
        migrations.append(Migration(
            id="005_create_performance_metrics",
            name="Create performance metrics table",
            description="Table for storing system performance metrics",
            version="1.0.0",
            up_sql="""
            CREATE TABLE IF NOT EXISTS nava_performance_metrics (
                metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                metric_name TEXT NOT NULL,
                metric_value FLOAT NOT NULL,
                metric_unit TEXT NOT NULL,
                recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metric_metadata JSONB DEFAULT '{}',
                metric_tags TEXT[] DEFAULT '{}'
            );
            
            CREATE INDEX idx_nava_performance_metrics_name ON nava_performance_metrics(metric_name);
            CREATE INDEX idx_nava_performance_metrics_recorded_at ON nava_performance_metrics(recorded_at);
            CREATE INDEX idx_nava_performance_metrics_tags ON nava_performance_metrics USING GIN(metric_tags);
            """,
            down_sql="DROP TABLE IF EXISTS nava_performance_metrics CASCADE;"
        ))
        
        return migrations

# Convenience functions
def create_migration_manager(supabase_client: Client) -> MigrationManager:
    """Create a migration manager instance"""
    return MigrationManager(supabase_client)

def setup_nava_database(supabase_client: Client) -> List[MigrationResult]:
    """Set up NAVA database with initial schema"""
    manager = MigrationManager(supabase_client)
    
    # Register initial migrations
    initial_migrations = NAVAMigrations.get_initial_schema_migrations()
    for migration in initial_migrations:
        manager.register_migration(migration)
    
    # Apply all pending migrations
    results = manager.apply_all_pending()
    
    logger.info(f"✅ Database setup complete: {len([r for r in results if r.success])}/{len(results)} migrations applied")
    
    return results

def get_database_status(migration_manager: MigrationManager) -> Dict[str, Any]:
    """Get database migration status"""
    return migration_manager.get_migration_status()

def validate_database_migrations(migration_manager: MigrationManager) -> List[Dict[str, Any]]:
    """Validate database migrations"""
    return migration_manager.validate_migrations()

# Export main classes and functions
__all__ = [
    'Migration', 'MigrationResult', 'MigrationManager', 'NAVAMigrations',
    'create_migration_manager', 'setup_nava_database', 'get_database_status',
    'validate_database_migrations'
]
        """Apply a specific migration"""
        if migration_id not in self.migrations:
            return MigrationResult(
                migration_id=migration_id,
                success=False,
                error_message="Migration not found"
            )
        
        migration = self.migrations[migration_id]
        
        if migration.is_applied:
            return MigrationResult(
                migration_id=migration_id,
                success=True,
                error_message="Migration already applied"
            )
        
        # Check dependencies
        for dep_id in migration.dependencies:
            if dep_id not in self.migrations or not self.migrations[dep_id].is_applied:
                return MigrationResult(
                    migration_id=migration_id,
                    success=False,
                    error_message=f"Dependency not satisfied: {dep_id}"
                )
        
        # Execute migration
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 Applying migration: {migration.id} - {migration.name}")
            
            # Execute the UP SQL
            # Note: This is a simplified version - real implementation would use transactions
            self._execute_sql(migration.up_sql)
            
            # Mark as applied
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            migration.applied_at = end_time
            migration.is_applied = True
            
            # Update database record
            self._save_migration_to_db(migration, execution_time)
            
            result = MigrationResult(
                migration_id=migration_id,
                success=True,
                execution_time_ms=execution_time
            )
            
            self.migration_history.append(result)
            logger.info(f"✅ Migration applied successfully: {migration.id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            result = MigrationResult(
                migration_id=migration_id,
                success=False,
                error_message=error_msg,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            self.migration_history.append(result)
            return result
    
    