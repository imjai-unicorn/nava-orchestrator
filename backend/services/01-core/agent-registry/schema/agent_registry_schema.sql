-- ============================================================================
-- NAVA Agent Registry - Supabase Database Schema
-- ============================================================================
-- Purpose: Database tables for Agent Registry service
-- Version: 1.0.0
-- Created: 2024
-- ============================================================================

-- Enable UUID extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable RLS (Row Level Security)
ALTER DATABASE postgres SET row_security = on;

-- ============================================================================
-- 1. AI SERVICES REGISTRY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_services (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Service identification
    service_id VARCHAR(50) UNIQUE NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    service_url VARCHAR(255) NOT NULL,
    service_port INTEGER NOT NULL,
    service_type VARCHAR(20) NOT NULL CHECK (service_type IN ('external_ai', 'local_ai', 'hybrid')),
    
    -- Service capabilities
    models JSONB NOT NULL DEFAULT '[]',
    capabilities JSONB NOT NULL DEFAULT '[]',
    
    -- Cost and performance
    cost_per_1k_tokens DECIMAL(10,6) DEFAULT 0.0,
    max_tokens INTEGER DEFAULT 2048,
    timeout_seconds INTEGER DEFAULT 30,
    priority INTEGER DEFAULT 5,
    max_concurrent INTEGER DEFAULT 10,
    
    -- Health and status
    status VARCHAR(20) DEFAULT 'offline' CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'offline', 'maintenance')),
    last_health_check TIMESTAMPTZ,
    response_time_ms DECIMAL(10,2) DEFAULT 0.0,
    success_rate DECIMAL(5,4) DEFAULT 0.0,
    error_count INTEGER DEFAULT 0,
    total_requests INTEGER DEFAULT 0,
    current_load INTEGER DEFAULT 0,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    is_active BOOLEAN DEFAULT true
);

-- Indexes for ai_services
CREATE INDEX idx_ai_services_service_id ON ai_services(service_id);
CREATE INDEX idx_ai_services_status ON ai_services(status);
CREATE INDEX idx_ai_services_type ON ai_services(service_type);
CREATE INDEX idx_ai_services_priority ON ai_services(priority);
CREATE INDEX idx_ai_services_health_check ON ai_services(last_health_check);
CREATE INDEX idx_ai_services_active ON ai_services(is_active);

-- ============================================================================
-- 2. SERVICE HEALTH HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS service_health_history (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Foreign key to ai_services
    service_id UUID NOT NULL REFERENCES ai_services(id) ON DELETE CASCADE,
    
    -- Health check data
    check_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,
    response_time_ms DECIMAL(10,2),
    error_message TEXT,
    health_score DECIMAL(3,2), -- 0.00 to 1.00
    
    -- Additional metrics
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    concurrent_requests INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for service_health_history
CREATE INDEX idx_health_history_service_id ON service_health_history(service_id);
CREATE INDEX idx_health_history_timestamp ON service_health_history(check_timestamp);
CREATE INDEX idx_health_history_status ON service_health_history(status);

-- Partition by month for performance
CREATE INDEX idx_health_history_monthly ON service_health_history(service_id, check_timestamp);

-- ============================================================================
-- 3. SERVICE SELECTION LOGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS service_selection_logs (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Selection context
    request_id VARCHAR(100),
    user_id UUID,
    session_id VARCHAR(100),
    
    -- Selection criteria
    requested_capability VARCHAR(50),
    model_preference VARCHAR(50),
    cost_priority BOOLEAN DEFAULT false,
    performance_priority BOOLEAN DEFAULT false,
    
    -- Selection result
    selected_service_id UUID REFERENCES ai_services(id),
    selection_reason TEXT,
    confidence_score DECIMAL(3,2),
    
    -- Failover information
    failover_chain JSONB DEFAULT '[]',
    fallover_used BOOLEAN DEFAULT false,
    
    -- Performance metrics
    selection_time_ms DECIMAL(10,2),
    total_request_time_ms DECIMAL(10,2),
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for service_selection_logs
CREATE INDEX idx_selection_logs_request_id ON service_selection_logs(request_id);
CREATE INDEX idx_selection_logs_user_id ON service_selection_logs(user_id);
CREATE INDEX idx_selection_logs_service_id ON service_selection_logs(selected_service_id);
CREATE INDEX idx_selection_logs_timestamp ON service_selection_logs(created_at);
CREATE INDEX idx_selection_logs_capability ON service_selection_logs(requested_capability);

-- ============================================================================
-- 4. LOAD BALANCING METRICS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS load_balancing_metrics (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Service reference
    service_id UUID NOT NULL REFERENCES ai_services(id) ON DELETE CASCADE,
    
    -- Time window (hourly aggregation)
    time_window TIMESTAMPTZ NOT NULL,
    
    -- Load metrics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(10,2),
    p95_response_time_ms DECIMAL(10,2),
    p99_response_time_ms DECIMAL(10,2),
    
    -- Concurrency metrics
    max_concurrent_requests INTEGER DEFAULT 0,
    avg_concurrent_requests DECIMAL(5,2),
    queue_wait_time_ms DECIMAL(10,2),
    
    -- Error analysis
    timeout_errors INTEGER DEFAULT 0,
    connection_errors INTEGER DEFAULT 0,
    server_errors INTEGER DEFAULT 0,
    client_errors INTEGER DEFAULT 0,
    
    -- Cost metrics
    total_tokens_processed INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0.0,
    
    -- Created timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure unique time windows per service
    UNIQUE(service_id, time_window)
);

-- Indexes for load_balancing_metrics
CREATE INDEX idx_load_metrics_service_id ON load_balancing_metrics(service_id);
CREATE INDEX idx_load_metrics_time_window ON load_balancing_metrics(time_window);
CREATE INDEX idx_load_metrics_service_time ON load_balancing_metrics(service_id, time_window);

-- ============================================================================
-- 5. CIRCUIT BREAKER EVENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Service reference
    service_id UUID NOT NULL REFERENCES ai_services(id) ON DELETE CASCADE,
    
    -- Event details
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('opened', 'closed', 'half_open', 'failure', 'success')),
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Circuit breaker state
    failure_count INTEGER,
    success_count INTEGER,
    failure_threshold INTEGER,
    success_threshold INTEGER,
    timeout_duration INTEGER,
    
    -- Event context
    error_message TEXT,
    response_time_ms DECIMAL(10,2),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for circuit_breaker_events
CREATE INDEX idx_circuit_events_service_id ON circuit_breaker_events(service_id);
CREATE INDEX idx_circuit_events_timestamp ON circuit_breaker_events(event_timestamp);
CREATE INDEX idx_circuit_events_type ON circuit_breaker_events(event_type);

-- ============================================================================
-- 6. SERVICE CONFIGURATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS service_configurations (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Service reference
    service_id UUID NOT NULL REFERENCES ai_services(id) ON DELETE CASCADE,
    
    -- Configuration data
    config_key VARCHAR(100) NOT NULL,
    config_value JSONB NOT NULL,
    config_type VARCHAR(20) DEFAULT 'general',
    
    -- Version control
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    
    -- Ensure unique active configs per service/key
    UNIQUE(service_id, config_key, is_active) WHERE is_active = true
);

-- Indexes for service_configurations
CREATE INDEX idx_service_configs_service_id ON service_configurations(service_id);
CREATE INDEX idx_service_configs_key ON service_configurations(config_key);
CREATE INDEX idx_service_configs_active ON service_configurations(is_active);

-- ============================================================================
-- 7. ALERTS AND NOTIFICATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS registry_alerts (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL,
    alert_level VARCHAR(20) NOT NULL CHECK (alert_level IN ('info', 'warning', 'error', 'critical')),
    alert_title VARCHAR(200) NOT NULL,
    alert_message TEXT NOT NULL,
    
    -- Service context (optional)
    service_id UUID REFERENCES ai_services(id),
    
    -- Alert status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved', 'ignored')),
    
    -- Timing
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    
    -- Notification tracking
    notification_sent BOOLEAN DEFAULT false,
    notification_channels JSONB DEFAULT '[]',
    
    -- User assignment
    assigned_to UUID,
    acknowledged_by UUID,
    resolved_by UUID,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for registry_alerts
CREATE INDEX idx_alerts_type ON registry_alerts(alert_type);
CREATE INDEX idx_alerts_level ON registry_alerts(alert_level);
CREATE INDEX idx_alerts_status ON registry_alerts(status);
CREATE INDEX idx_alerts_service_id ON registry_alerts(service_id);
CREATE INDEX idx_alerts_triggered ON registry_alerts(triggered_at);

-- ============================================================================
-- 8. PERFORMANCE ANALYTICS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_analytics (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Time dimension
    date_key DATE NOT NULL,
    hour_key INTEGER NOT NULL CHECK (hour_key >= 0 AND hour_key <= 23),
    
    -- Service reference
    service_id UUID REFERENCES ai_services(id) ON DELETE CASCADE,
    
    -- Performance metrics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    avg_response_time DECIMAL(10,2),
    min_response_time DECIMAL(10,2),
    max_response_time DECIMAL(10,2),
    p50_response_time DECIMAL(10,2),
    p95_response_time DECIMAL(10,2),
    p99_response_time DECIMAL(10,2),
    
    -- Availability metrics
    uptime_percentage DECIMAL(5,2),
    downtime_minutes INTEGER DEFAULT 0,
    
    -- Cost metrics
    total_cost DECIMAL(10,6) DEFAULT 0.0,
    cost_per_request DECIMAL(10,6) DEFAULT 0.0,
    
    -- Quality metrics
    avg_quality_score DECIMAL(3,2),
    
    -- Created timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(date_key, hour_key, service_id)
);

-- Indexes for performance_analytics
CREATE INDEX idx_perf_analytics_date ON performance_analytics(date_key);
CREATE INDEX idx_perf_analytics_service ON performance_analytics(service_id);
CREATE INDEX idx_perf_analytics_date_service ON performance_analytics(date_key, service_id);

-- ============================================================================
-- 9. AUDIT LOGS TABLE (Agent Registry Specific)
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_registry_audit_logs (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Audit context
    user_id UUID,
    session_id VARCHAR(100),
    request_id VARCHAR(100),
    
    -- Action details
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    
    -- Data changes
    old_values JSONB,
    new_values JSONB,
    
    -- Request details
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(255),
    http_method VARCHAR(10),
    
    -- Result
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    -- Timestamp
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for agent_registry_audit_logs
CREATE INDEX idx_registry_audit_user_id ON agent_registry_audit_logs(user_id);
CREATE INDEX idx_registry_audit_action ON agent_registry_audit_logs(action);
CREATE INDEX idx_registry_audit_resource ON agent_registry_audit_logs(resource_type);
CREATE INDEX idx_registry_audit_timestamp ON agent_registry_audit_logs(timestamp);

-- ============================================================================
-- 10. SYSTEM CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS registry_system_config (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Configuration
    config_category VARCHAR(50) NOT NULL,
    config_key VARCHAR(100) NOT NULL,
    config_value JSONB NOT NULL,
    
    -- Description and validation
    description TEXT,
    data_type VARCHAR(20) DEFAULT 'string',
    validation_rules JSONB DEFAULT '{}',
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    is_sensitive BOOLEAN DEFAULT false,
    
    -- Version control
    version INTEGER DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID,
    
    -- Ensure unique active configs
    UNIQUE(config_category, config_key, is_active) WHERE is_active = true
);

-- Indexes for registry_system_config
CREATE INDEX idx_system_config_category ON registry_system_config(config_category);
CREATE INDEX idx_system_config_key ON registry_system_config(config_key);
CREATE INDEX idx_system_config_active ON registry_system_config(is_active);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Healthy Services
CREATE OR REPLACE VIEW healthy_services AS
SELECT 
    s.*,
    CASE 
        WHEN s.status = 'healthy' THEN 1
        WHEN s.status = 'degraded' THEN 0.5
        ELSE 0
    END as health_weight
FROM ai_services s 
WHERE s.is_active = true 
AND s.status IN ('healthy', 'degraded')
ORDER BY s.priority, s.current_load;

-- View: Service Performance Summary
CREATE OR REPLACE VIEW service_performance_summary AS
SELECT 
    s.service_id,
    s.service_name,
    s.status,
    s.response_time_ms,
    s.success_rate,
    s.current_load,
    s.max_concurrent,
    ROUND((s.current_load::DECIMAL / s.max_concurrent) * 100, 2) as load_percentage,
    s.total_requests,
    s.error_count,
    s.last_health_check,
    EXTRACT(EPOCH FROM (NOW() - s.last_health_check))/60 as minutes_since_check
FROM ai_services s
WHERE s.is_active = true
ORDER BY s.priority, s.response_time_ms;

-- View: Alert Summary
CREATE OR REPLACE VIEW alert_summary AS
SELECT 
    alert_level,
    status,
    COUNT(*) as alert_count,
    MIN(triggered_at) as oldest_alert,
    MAX(triggered_at) as newest_alert
FROM registry_alerts 
WHERE status IN ('active', 'acknowledged')
GROUP BY alert_level, status
ORDER BY 
    CASE alert_level 
        WHEN 'critical' THEN 1 
        WHEN 'error' THEN 2 
        WHEN 'warning' THEN 3 
        WHEN 'info' THEN 4 
    END;

-- View: Daily Performance Stats
CREATE OR REPLACE VIEW daily_performance_stats AS
SELECT 
    date_key,
    service_id,
    SUM(total_requests) as daily_requests,
    SUM(successful_requests) as daily_successful,
    SUM(failed_requests) as daily_failed,
    ROUND(AVG(avg_response_time), 2) as avg_response_time,
    ROUND(AVG(uptime_percentage), 2) as avg_uptime,
    SUM(total_cost) as daily_cost
FROM performance_analytics
GROUP BY date_key, service_id
ORDER BY date_key DESC, service_id;

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Trigger: Auto-update updated_at for ai_services
CREATE TRIGGER update_ai_services_updated_at 
    BEFORE UPDATE ON ai_services 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for service_configurations
CREATE TRIGGER update_service_configurations_updated_at 
    BEFORE UPDATE ON service_configurations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for registry_system_config
CREATE TRIGGER update_registry_system_config_updated_at 
    BEFORE UPDATE ON registry_system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function: Calculate service health score
CREATE OR REPLACE FUNCTION calculate_health_score(
    p_response_time DECIMAL,
    p_success_rate DECIMAL,
    p_error_count INTEGER,
    p_total_requests INTEGER
) RETURNS DECIMAL AS $
DECLARE
    health_score DECIMAL := 1.0;
    response_penalty DECIMAL := 0.0;
    error_penalty DECIMAL := 0.0;
BEGIN
    -- Response time penalty (0-0.3 penalty for >3000ms)
    IF p_response_time > 3000 THEN
        response_penalty := LEAST(0.3, (p_response_time - 3000) / 10000);
    END IF;
    
    -- Error rate penalty (0-0.5 penalty based on error rate)
    IF p_total_requests > 0 THEN
        error_penalty := (p_error_count::DECIMAL / p_total_requests) * 0.5;
    END IF;
    
    -- Success rate component (0.5 weight)
    health_score := p_success_rate * 0.5;
    
    -- Response time component (0.3 weight)
    health_score := health_score + (1.0 - response_penalty) * 0.3;
    
    -- Reliability component (0.2 weight)
    health_score := health_score + (1.0 - error_penalty) * 0.2;
    
    RETURN GREATEST(0.0, LEAST(1.0, health_score));
END;
$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default system configurations
INSERT INTO registry_system_config (config_category, config_key, config_value, description) VALUES
('health_monitoring', 'check_interval_seconds', '30', 'Health check interval in seconds'),
('health_monitoring', 'timeout_seconds', '10', 'Health check timeout in seconds'),
('health_monitoring', 'max_failures', '5', 'Maximum consecutive failures before marking unhealthy'),
('load_balancing', 'strategy', '"priority_based"', 'Default load balancing strategy'),
('load_balancing', 'max_concurrent_per_service', '10', 'Maximum concurrent requests per service'),
('circuit_breaker', 'failure_threshold', '5', 'Circuit breaker failure threshold'),
('circuit_breaker', 'success_threshold', '3', 'Circuit breaker success threshold'),
('circuit_breaker', 'timeout_seconds', '60', 'Circuit breaker timeout in seconds'),
('alerting', 'response_time_threshold_ms', '3000', 'Alert threshold for response time'),
('alerting', 'error_rate_threshold', '0.05', 'Alert threshold for error rate'),
('caching', 'ttl_seconds', '300', 'Cache TTL in seconds'),
('caching', 'max_size', '1000', 'Maximum cache size');

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE ai_services ENABLE ROW LEVEL SECURITY;
ALTER TABLE service_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE registry_system_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_registry_audit_logs ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies (adjust based on your authentication system)

-- Policy: Public read for ai_services (for health monitoring)
CREATE POLICY "ai_services_read" ON ai_services
    FOR SELECT USING (true);

-- Policy: Authenticated users can insert/update ai_services
CREATE POLICY "ai_services_write" ON ai_services
    FOR ALL USING (auth.role() = 'authenticated');

-- Policy: Service configurations readable by authenticated users
CREATE POLICY "service_configs_read" ON service_configurations
    FOR SELECT USING (auth.role() = 'authenticated');

-- Policy: System configuration requires admin role
CREATE POLICY "system_config_admin" ON registry_system_config
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- PERFORMANCE OPTIMIZATIONS
-- ============================================================================

-- Partitioning for large tables (monthly partitions)
-- Note: This would be set up separately for high-volume deployments

-- Automatic cleanup of old data (adjust retention as needed)
-- CREATE OR REPLACE FUNCTION cleanup_old_data() RETURNS void AS $
-- BEGIN
--     -- Clean up health history older than 30 days
--     DELETE FROM service_health_history 
--     WHERE check_timestamp < NOW() - INTERVAL '30 days';
--     
--     -- Clean up selection logs older than 90 days
--     DELETE FROM service_selection_logs 
--     WHERE created_at < NOW() - INTERVAL '90 days';
--     
--     -- Clean up resolved alerts older than 7 days
--     DELETE FROM registry_alerts 
--     WHERE status = 'resolved' AND resolved_at < NOW() - INTERVAL '7 days';
-- END;
-- $ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions for the application user
-- Adjust these based on your Supabase setup

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO anon;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- ============================================================================
-- SCHEMA COMPLETION
-- ============================================================================

-- Add comments for documentation
COMMENT ON TABLE ai_services IS 'Registry of all AI services available to NAVA';
COMMENT ON TABLE service_health_history IS 'Historical health check data for services';
COMMENT ON TABLE service_selection_logs IS 'Log of service selection decisions';
COMMENT ON TABLE load_balancing_metrics IS 'Load balancing and performance metrics';
COMMENT ON TABLE circuit_breaker_events IS 'Circuit breaker state change events';
COMMENT ON TABLE performance_analytics IS 'Aggregated performance analytics data';
COMMENT ON TABLE registry_alerts IS 'System alerts and notifications';
COMMENT ON TABLE agent_registry_audit_logs IS 'Audit trail for Agent Registry operations';

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(10) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES 
('1.0.0', 'Initial Agent Registry schema with core tables and functions');

-- ============================================================================
-- SETUP COMPLETE
-- ============================================================================

-- Verify schema creation
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN (
    'ai_services',
    'service_health_history', 
    'service_selection_logs',
    'load_balancing_metrics',
    'circuit_breaker_events',
    'performance_analytics',
    'registry_alerts',
    'agent_registry_audit_logs',
    'service_configurations',
    'registry_system_config'
)
ORDER BY tablename;