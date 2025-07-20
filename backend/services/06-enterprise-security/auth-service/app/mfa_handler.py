# backend/services/06-enterprise-security/auth-service/app/mfa_handler.py
"""
Multi-Factor Authentication Handler
TOTP, SMS, and backup codes
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict
import pyotp
import qrcode
import io
import base64
import secrets
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Pydantic models
class MFASetupRequest(BaseModel):
    user_id: str

class MFASetupResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: List[str]
    setup_url: str

class MFAVerifyRequest(BaseModel):
    user_id: str
    code: str
    backup_code: Optional[str] = None

class MFAVerifyResponse(BaseModel):
    valid: bool
    message: str

class MFAStatusResponse(BaseModel):
    enabled: bool
    backup_codes_remaining: int
    last_used: Optional[datetime]

class MFAHandler:
    def __init__(self):
        self.mfa_secrets = {}  # In production, use database
        self.backup_codes = {}
        self.used_codes = {}
        logger.info("🔐 MFA Handler initialized")

    def generate_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        self.mfa_secrets[user_id] = {
            "secret": secret,
            "enabled": False,
            "created_at": datetime.now()
        }
        return secret

    def generate_backup_codes(self, user_id: str, count: int = 8) -> List[str]:
        """Generate backup codes for user"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8-character hex codes
            codes.append(code)
        
        self.backup_codes[user_id] = {
            "codes": codes,
            "used": [],
            "created_at": datetime.now()
        }
        
        return codes

    def generate_qr_code(self, user_id: str, secret: str, issuer: str = "NAVA Enterprise") -> str:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        
        return base64.b64encode(img_data).decode()

    def verify_totp_code(self, user_id: str, code: str) -> bool:
        """Verify TOTP code"""
        user_mfa = self.mfa_secrets.get(user_id)
        if not user_mfa:
            return False

        secret = user_mfa["secret"]
        totp = pyotp.TOTP(secret)
        
        # Allow for time drift (30 seconds before/after)
        return totp.verify(code, valid_window=1)

    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code"""
        user_backup = self.backup_codes.get(user_id)
        if not user_backup:
            return False

        code = code.upper().strip()
        
        # Check if code exists and hasn't been used
        if code in user_backup["codes"] and code not in user_backup["used"]:
            user_backup["used"].append(code)
            logger.info(f"✅ Backup code used for user: {user_id}")
            return True

        return False

    def enable_mfa(self, user_id: str) -> bool:
        """Enable MFA for user"""
        if user_id in self.mfa_secrets:
            self.mfa_secrets[user_id]["enabled"] = True
            logger.info(f"✅ MFA enabled for user: {user_id}")
            return True
        return False

    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user"""
        if user_id in self.mfa_secrets:
            self.mfa_secrets[user_id]["enabled"] = False
            logger.info(f"⚠️ MFA disabled for user: {user_id}")
            return True
        return False

    def get_mfa_status(self, user_id: str) -> Dict:
        """Get MFA status for user"""
        user_mfa = self.mfa_secrets.get(user_id, {})
        user_backup = self.backup_codes.get(user_id, {})
        
        remaining_codes = 0
        if user_backup:
            total_codes = len(user_backup.get("codes", []))
            used_codes = len(user_backup.get("used", []))
            remaining_codes = total_codes - used_codes

        return {
            "enabled": user_mfa.get("enabled", False),
            "backup_codes_remaining": remaining_codes,
            "last_used": user_mfa.get("last_used")
        }

# Initialize MFA handler
mfa_handler = MFAHandler()

@router.post("/setup", response_model=MFASetupResponse)
async def setup_mfa(request: MFASetupRequest):
    """Setup MFA for user"""
    user_id = request.user_id
    
    # Generate secret and backup codes
    secret = mfa_handler.generate_secret(user_id)
    backup_codes = mfa_handler.generate_backup_codes(user_id)
    
    # Generate QR code
    qr_code = mfa_handler.generate_qr_code(user_id, secret)
    
    # Generate setup URL
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_id,
        issuer_name="NAVA Enterprise"
    )
    
    logger.info(f"✅ MFA setup generated for user: {user_id}")
    
    return MFASetupResponse(
        secret=secret,
        qr_code=qr_code,
        backup_codes=backup_codes,
        setup_url=totp_uri
    )

@router.post("/verify", response_model=MFAVerifyResponse)
async def verify_mfa(request: MFAVerifyRequest):
    """Verify MFA code"""
    user_id = request.user_id
    
    # Try TOTP code first
    if request.code:
        if mfa_handler.verify_totp_code(user_id, request.code):
            # Update last used time
            if user_id in mfa_handler.mfa_secrets:
                mfa_handler.mfa_secrets[user_id]["last_used"] = datetime.now()
            
            return MFAVerifyResponse(
                valid=True,
                message="TOTP code verified successfully"
            )
    
    # Try backup code if TOTP failed
    if request.backup_code:
        if mfa_handler.verify_backup_code(user_id, request.backup_code):
            return MFAVerifyResponse(
                valid=True,
                message="Backup code verified successfully"
            )
    
    logger.warning(f"❌ MFA verification failed for user: {user_id}")
    return MFAVerifyResponse(
        valid=False,
        message="Invalid verification code"
    )

@router.post("/enable")
async def enable_mfa(request: MFASetupRequest):
    """Enable MFA for user"""
    if mfa_handler.enable_mfa(request.user_id):
        return {"message": "MFA enabled successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MFA setup not found for user"
        )

@router.post("/disable")
async def disable_mfa(request: MFASetupRequest):
    """Disable MFA for user"""
    if mfa_handler.disable_mfa(request.user_id):
        return {"message": "MFA disabled successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MFA not found for user"
        )

@router.get("/status/{user_id}", response_model=MFAStatusResponse)
async def get_mfa_status(user_id: str):
    """Get MFA status for user"""
    status_data = mfa_handler.get_mfa_status(user_id)
    
    return MFAStatusResponse(
        enabled=status_data["enabled"],
        backup_codes_remaining=status_data["backup_codes_remaining"],
        last_used=status_data["last_used"]
    )

@router.post("/regenerate-backup-codes")
async def regenerate_backup_codes(request: MFASetupRequest):
    """Regenerate backup codes for user"""
    user_id = request.user_id
    
    # Check if user has MFA setup
    if user_id not in mfa_handler.mfa_secrets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MFA not setup for user"
        )
    
    # Generate new backup codes
    new_codes = mfa_handler.generate_backup_codes(user_id)
    
    logger.info(f"✅ Backup codes regenerated for user: {user_id}")
    
    return {
        "message": "Backup codes regenerated successfully",
        "backup_codes": new_codes
    }

# Export router
mfa_router = router