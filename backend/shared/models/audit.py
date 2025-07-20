# backend/shared/models/audit.py
"""
Shared Audit Models for NAVA
Comprehensive audit trail and compliance tracking models
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json

class AuditEventType(Enum):
    """Types of audit events"""
    USER_ACTION = "user_action"
    SYSTEM_ACTION = "system_action"
    AI_REQUEST = "ai_request"
    AI_RESPONSE = "ai_response"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_EVENT = "error_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    CUSTOM = "custom"

@dataclass
class AuditEvent:
    """Individual audit event"""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service: Optional[str] = None
    endpoint: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.LOW
    success: bool = True
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[ComplianceStandard] = field(default_factory=list)
    retention_days: int = 2555  # Default 7 years for compliance

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "service": self.service,
            "endpoint": self.endpoint,
            "action": self.action,
            "resource": self.resource,
            "severity": self.severity.value,
            "success": self.success,
            "error_message": self.error_message,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "metadata": self.metadata,
            "compliance_tags": [tag.value for tag in self.compliance_tags],
            "retention_days": self.retention_days
        }

    def to_json(self) -> str:
        """Convert audit event to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary"""
        return cls(
            id=data["id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            service=data.get("service"),
            endpoint=data.get("endpoint"),
            action=data.get("action"),
            resource=data.get("resource"),
            severity=AuditSeverity(data.get("severity", "low")),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            request_data=data.get("request_data"),
            response_data=data.get("response_data"),
            metadata=data.get("metadata", {}),
            compliance_tags=[ComplianceStandard(tag) for tag in data.get("compliance_tags", [])],
            retention_days=data.get("retention_days", 2555)
        )

@dataclass
class ComplianceReport:
    """Compliance report data"""
    id: str
    standard: ComplianceStandard
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_events: int
    compliant_events: int
    non_compliant_events: int
    compliance_score: float  # 0.0 to 1.0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert compliance report to dictionary"""
        return {
            "id": self.id,
            "standard": self.standard.value,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_events": self.total_events,
            "compliant_events": self.compliant_events,
            "non_compliant_events": self.non_compliant_events,
            "compliance_score": self.compliance_score,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }

@dataclass
class AuditQuery:
    """Audit query parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    user_id: Optional[str] = None
    service: Optional[str] = None
    severity: Optional[AuditSeverity] = None
    success: Optional[bool] = None
    compliance_standards: Optional[List[ComplianceStandard]] = None
    search_text: Optional[str] = None
    limit: int = 1000
    offset: int = 0
    sort_by: str = "timestamp"
    sort_order: str = "desc"  # desc or asc

@dataclass
class AuditSummary:
    """Audit summary statistics"""
    period_start: datetime
    period_end: datetime
    total_events: int
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    events_by_service: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    top_users: List[Dict[str, Any]] = field(default_factory=list)
    top_resources: List[Dict[str, Any]] = field(default_factory=list)
    compliance_scores: Dict[str, float] = field(default_factory=dict)
    security_incidents: int = 0
    data_access_events: int = 0
    authentication_events: int = 0

@dataclass 
class RetentionPolicy:
    """Data retention policy"""
    id: str
    name: str
    description: str
    event_types: List[AuditEventType]
    compliance_standards: List[ComplianceStandard]
    retention_days: int
    archive_after_days: Optional[int] = None
    delete_after_days: Optional[int] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def applies_to_event(self, event: AuditEvent) -> bool:
        """Check if retention policy applies to an event"""
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check compliance standards
        if self.compliance_standards:
            if not any(standard in event.compliance_tags for standard in self.compliance_standards):
                return False
        
        return True

# Utility functions for audit models
def create_audit_event(event_type: AuditEventType, 
                      action: str,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      service: Optional[str] = None,
                      **kwargs) -> AuditEvent:
    """Create a new audit event with common fields"""
    from uuid import uuid4
    
    return AuditEvent(
        id=str(uuid4()),
        event_type=event_type,
        timestamp=datetime.now(),
        user_id=user_id,
        session_id=session_id,
        service=service,
        action=action,
        **kwargs
    )

def create_user_action_audit(action: str,
                           user_id: str,
                           session_id: Optional[str] = None,
                           resource: Optional[str] = None,
                           **kwargs) -> AuditEvent:
    """Create user action audit event"""
    return create_audit_event(
        AuditEventType.USER_ACTION,
        action=action,
        user_id=user_id,
        session_id=session_id,
        resource=resource,
        **kwargs
    )

def create_ai_request_audit(model: str,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          prompt: Optional[str] = None,
                          **kwargs) -> AuditEvent:
    """Create AI request audit event"""
    metadata = kwargs.get("metadata", {})
    metadata.update({
        "ai_model": model,
        "prompt_length": len(prompt) if prompt else 0
    })
    
    return create_audit_event(
        AuditEventType.AI_REQUEST,
        action="ai_request",
        user_id=user_id,
        session_id=session_id,
        resource=f"ai_model:{model}",
        metadata=metadata,
        **kwargs
    )

def create_data_access_audit(resource: str,
                           action: str,
                           user_id: Optional[str] = None,
                           **kwargs) -> AuditEvent:
    """Create data access audit event"""
    return create_audit_event(
        AuditEventType.DATA_ACCESS,
        action=action,
        user_id=user_id,
        resource=resource,
        severity=AuditSeverity.MEDIUM,
        compliance_tags=[ComplianceStandard.GDPR],
        **kwargs
    )

def create_security_audit(action: str,
                        severity: AuditSeverity = AuditSeverity.HIGH,
                        user_id: Optional[str] = None,
                        **kwargs) -> AuditEvent:
    """Create security audit event"""
    return create_audit_event(
        AuditEventType.SECURITY_EVENT,
        action=action,
        user_id=user_id,
        severity=severity,
        compliance_tags=[ComplianceStandard.ISO27001],
        **kwargs
    )

def create_compliance_audit(standard: ComplianceStandard,
                          action: str,
                          **kwargs) -> AuditEvent:
    """Create compliance audit event"""
    return create_audit_event(
        AuditEventType.COMPLIANCE_EVENT,
        action=action,
        compliance_tags=[standard],
        **kwargs
    )

# Compliance-specific audit helpers
def create_gdpr_audit(action: str, 
                     data_subject: str,
                     user_id: Optional[str] = None,
                     **kwargs) -> AuditEvent:
    """Create GDPR-specific audit event"""
    metadata = kwargs.get("metadata", {})
    metadata.update({
        "data_subject": data_subject,
        "gdpr_lawful_basis": kwargs.get("lawful_basis", "legitimate_interest")
    })
    
    return create_audit_event(
        AuditEventType.COMPLIANCE_EVENT,
        action=f"gdpr_{action}",
        user_id=user_id,
        compliance_tags=[ComplianceStandard.GDPR],
        retention_days=2555,  # 7 years for GDPR
        metadata=metadata,
        **kwargs
    )

def create_sox_audit(action: str,
                    financial_data: bool = False,
                    user_id: Optional[str] = None,
                    **kwargs) -> AuditEvent:
    """Create SOX-specific audit event"""
    metadata = kwargs.get("metadata", {})
    metadata.update({
        "financial_data_involved": financial_data,
        "sox_control_activity": kwargs.get("control_activity", "general")
    })
    
    return create_audit_event(
        AuditEventType.COMPLIANCE_EVENT,
        action=f"sox_{action}",
        user_id=user_id,
        compliance_tags=[ComplianceStandard.SOX],
        retention_days=2555,  # 7 years for SOX
        severity=AuditSeverity.HIGH if financial_data else AuditSeverity.MEDIUM,
        metadata=metadata,
        **kwargs
    )