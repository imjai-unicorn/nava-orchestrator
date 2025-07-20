#!/usr/bin/env python3
"""
File Validation Script - ตรวจสอบไฟล์ว่าสมบูรณ์
Run: python validate_files.py
"""

import os
import sys
from pathlib import Path
import importlib.util
import ast

class FileValidator:
    def __init__(self, base_path="E:/nava-projects"):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        self.success = []
        
    def validate_file_exists(self, file_path):
        """ตรวจสอบว่าไฟล์มีอยู่"""
        full_path = self.base_path / file_path
        if full_path.exists():
            self.success.append(f"✅ {file_path} - EXISTS")
            return True
        else:
            self.errors.append(f"❌ {file_path} - MISSING")
            return False
    
    def validate_python_syntax(self, file_path):
        """ตรวจสอบ Python syntax"""
        full_path = self.base_path / file_path
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file is not empty
            if not content.strip():
                self.warnings.append(f"⚠️ {file_path} - EMPTY FILE")
                return False
                
            # Parse AST to check syntax
            ast.parse(content)
            self.success.append(f"✅ {file_path} - VALID PYTHON SYNTAX")
            return True
            
        except SyntaxError as e:
            self.errors.append(f"❌ {file_path} - SYNTAX ERROR: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ {file_path} - ERROR: {e}")
            return False
    
    def validate_imports(self, file_path, required_imports):
        """ตรวจสอบ required imports"""
        full_path = self.base_path / file_path
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for import_name in required_imports:
                if import_name not in content:
                    self.warnings.append(f"⚠️ {file_path} - MISSING IMPORT: {import_name}")
                    
            return True
        except Exception as e:
            self.errors.append(f"❌ {file_path} - IMPORT CHECK ERROR: {e}")
            return False
    
    def validate_class_methods(self, file_path, required_classes):
        """ตรวจสอบ required classes และ methods"""
        full_path = self.base_path / file_path
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Find all class definitions
            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes[node.name] = methods
            
            # Check required classes
            for class_name, required_methods in required_classes.items():
                if class_name not in classes:
                    self.errors.append(f"❌ {file_path} - MISSING CLASS: {class_name}")
                    continue
                    
                # Check required methods
                for method_name in required_methods:
                    if method_name not in classes[class_name]:
                        self.warnings.append(f"⚠️ {file_path} - MISSING METHOD: {class_name}.{method_name}")
                        
            return True
            
        except Exception as e:
            self.errors.append(f"❌ {file_path} - CLASS CHECK ERROR: {e}")
            return False
    
    def validate_javascript_syntax(self, file_path):
        """ตรวจสอบ JavaScript syntax (basic)"""
        full_path = self.base_path / file_path
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic syntax checks
            if not content.strip():
                self.warnings.append(f"⚠️ {file_path} - EMPTY FILE")
                return False
                
            # Check for basic React/JS patterns
            if 'import React' in content or 'import' in content:
                self.success.append(f"✅ {file_path} - VALID JAVASCRIPT/REACT")
                return True
            else:
                self.warnings.append(f"⚠️ {file_path} - NO IMPORTS FOUND")
                return False
                
        except Exception as e:
            self.errors.append(f"❌ {file_path} - JS CHECK ERROR: {e}")
            return False
    
    def run_validation(self):
        """รันการตรวจสอบทั้งหมด"""
        print("🔍 Starting File Validation...")
        print("=" * 50)
        
        # 1. Enhanced Circuit Breaker
        print("\n📁 1. Enhanced Circuit Breaker")
        file_path = "backend/services/shared/common/enhanced_circuit_breaker.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'import asyncio',
                'import logging',
                'from typing import',
                'from enum import Enum',
                'class CircuitState',
                'class EnhancedCircuitBreaker'
            ])
            self.validate_class_methods(file_path, {
                'EnhancedCircuitBreaker': [
                    'execute_with_circuit_breaker',
                    'execute_with_failover',
                    'record_success',
                    'record_failure',
                    'get_system_health'
                ]
            })
        
        # 2. GPT Client
        print("\n📁 2. GPT Client")
        file_path = "backend/services/03-external-ai/gpt-client/app/gpt_client.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'from openai import AsyncOpenAI',
                'import asyncio',
                'import logging',
                'class GPTClient'
            ])
            self.validate_class_methods(file_path, {
                'GPTClient': [
                    'create_chat_completion',
                    'create_completion',
                    'health_check',
                    'get_service_info'
                ]
            })
        
        # 3. Claude Client
        print("\n📁 3. Claude Client")
        file_path = "backend/services/03-external-ai/claude-client/app/claude_client.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'import anthropic',
                'import asyncio',
                'import logging',
                'class ClaudeClient'
            ])
            self.validate_class_methods(file_path, {
                'ClaudeClient': [
                    'create_chat_completion',
                    'create_completion',
                    'create_reasoning_enhanced',
                    'health_check'
                ]
            })
        
        # 4. Gemini Client
        print("\n📁 4. Gemini Client")
        file_path = "backend/services/03-external-ai/gemini-client/app/gemini_client.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'import google.generativeai as genai',
                'import asyncio',
                'import logging',
                'class GeminiClient'
            ])
            self.validate_class_methods(file_path, {
                'GeminiClient': [
                    'create_chat_completion',
                    'create_completion',
                    'create_multimodal_completion',
                    'health_check'
                ]
            })
        
        # 5. Performance Utils
        print("\n📁 5. Performance Utils")
        file_path = "backend/services/shared/common/performance_utils.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'import psutil',
                'import threading',
                'from collections import defaultdict',
                'class PerformanceMonitor'
            ])
            self.validate_class_methods(file_path, {
                'PerformanceMonitor': [
                    'record_request',
                    'get_metrics',
                    'get_system_metrics'
                ]
            })
        
        # 6. Shared Models
        print("\n📁 6. Shared Models")
        file_path = "backend/services/shared/models/base.py"
        if self.validate_file_exists(file_path):
            self.validate_python_syntax(file_path)
            self.validate_imports(file_path, [
                'from pydantic import BaseModel',
                'from typing import Dict, Any',
                'from enum import Enum',
                'class ServiceStatus'
            ])
        
        # 7. Frontend Components
        print("\n📁 7. Frontend Components")
        components = [
            "frontend/customer-chat/src/components/AgentSelector.jsx",
            "frontend/customer-chat/src/components/LoadingSpinner.jsx",
            "frontend/customer-chat/src/components/FeedbackForm.jsx",
            "frontend/customer-chat/src/services/cache.js"
        ]
        
        for component in components:
            if self.validate_file_exists(component):
                self.validate_javascript_syntax(component)
        
        # 8. Existing Files Check
        print("\n📁 8. Existing Core Files")
        existing_files = [
            "backend/services/01-core/nava-logic-controller/app/core/controller.py",
            "backend/services/01-core/nava-logic-controller/app/service/logic_orchestrator.py",
            "backend/services/05-enhanced-intelligence/decision-engine/app/enhanced_decision_engine.py",
            "backend/services/05-enhanced-intelligence/quality-service/app/quality_validator.py",
            "backend/services/05-enhanced-intelligence/slf-framework/app/slf_enhancer.py",
            "backend/services/05-enhanced-intelligence/cache-engine/app/cache_manager.py"
        ]
        
        for file_path in existing_files:
            self.validate_file_exists(file_path)
            if self.base_path / file_path:
                self.validate_python_syntax(file_path)
        
        # Print Results
        self.print_results()
        
        return len(self.errors) == 0
    
    def print_results(self):
        """แสดงผลลัพธ์"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION RESULTS")
        print("=" * 50)
        
        if self.success:
            print(f"\n✅ SUCCESS ({len(self.success)} items):")
            for item in self.success:
                print(f"  {item}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)} items):")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)} items):")
            for item in self.errors:
                print(f"  {item}")
        
        print(f"\n📈 SUMMARY:")
        print(f"  ✅ Success: {len(self.success)}")
        print(f"  ⚠️ Warnings: {len(self.warnings)}")
        print(f"  ❌ Errors: {len(self.errors)}")
        
        if len(self.errors) == 0:
            print("\n🎉 ALL FILES VALIDATED SUCCESSFULLY!")
        else:
            print("\n❌ VALIDATION FAILED - Please fix errors above")

def main():
    # ใช้ path จริงของคุณ
    validator = FileValidator("E:/nava-projects")  # เปลี่ยนเป็น path ของคุณ
    
    success = validator.run_validation()
    
    if success:
        print("\n🚀 Ready for Local Testing!")
        return 0
    else:
        print("\n🔧 Please fix errors before testing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
