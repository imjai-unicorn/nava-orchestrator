# NAVA Testing Dashboard - Streamlit App
# Fix test issues and provide interactive testing interface

import streamlit as st
import sys
import os
from pathlib import Path
import traceback
import json
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NAVA Testing Dashboard",
    page_icon="🧪",
    layout="wide"
)

def setup_paths():
    """Setup Python paths for imports"""
    try:
        # Get current directory and navigate to project structure
        current_dir = Path(__file__).parent if hasattr(__file__, '__file__') else Path('.')
        
        # Assume we're running from tests/ directory or root
        if current_dir.name == 'tests':
            app_path = current_dir.parent / 'app'
        else:
            app_path = current_dir / 'app'
        
        if app_path.exists() and str(app_path) not in sys.path:
            sys.path.insert(0, str(app_path))
            return str(app_path)
        
        # Alternative path setup for different locations
        possible_paths = [
            Path('app'),
            Path('../app'),
            Path('./nava-logic-controller/app'),
            Path('../nava-logic-controller/app')
        ]
        
        for path in possible_paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
                return str(path)
                
        return None
        
    except Exception as e:
        st.error(f"Path setup error: {e}")
        return None

def test_imports():
    """Test if we can import the decision engine"""
    try:
        from core.decision_engine import EnhancedDecisionEngine
        return True, "Success"
    except ImportError as e:
        return False, f"Import Error: {e}"
    except Exception as e:
        return False, f"Other Error: {e}"

def test_basic_functionality():
    """Test basic decision engine functionality"""
    try:
        from core.decision_engine import EnhancedDecisionEngine
        
        engine = EnhancedDecisionEngine()
        
        # Test model selection
        result = engine.select_model("test message")
        
        if not isinstance(result, tuple) or len(result) != 3:
            return False, f"Invalid result format: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
        
        model, confidence, reasoning = result
        
        # Validate result components
        if not isinstance(model, str):
            return False, f"Model should be string, got: {type(model)}"
        
        if model not in ["gpt", "claude", "gemini"]:
            return False, f"Unknown model: {model}"
        
        if not isinstance(confidence, (int, float)):
            return False, f"Confidence should be numeric, got: {type(confidence)}"
        
        if not (0 <= confidence <= 1):
            return False, f"Confidence should be 0-1, got: {confidence}"
        
        if not isinstance(reasoning, dict):
            return False, f"Reasoning should be dict, got: {type(reasoning)}"
        
        return True, {
            "model": model,
            "confidence": confidence,
            "reasoning_keys": list(reasoning.keys())
        }
        
    except Exception as e:
        return False, f"Error: {e}\n{traceback.format_exc()}"

def test_feedback_system():
    """Test feedback system functionality"""
    try:
        from core.decision_engine import EnhancedDecisionEngine
        
        engine = EnhancedDecisionEngine()
        
        # Test feedback submission
        engine.update_user_feedback("test_123", "gpt", "conversation", 4.0)
        
        # Test stats retrieval
        stats = engine.get_feedback_stats()
        
        if not isinstance(stats, dict):
            return False, f"Stats should be dict, got: {type(stats)}"
        
        if "feedback_summary" not in stats:
            return False, "Missing feedback_summary in stats"
        
        return True, stats["feedback_summary"]
        
    except Exception as e:
        return False, f"Error: {e}\n{traceback.format_exc()}"

def test_multiple_selections():
    """Test multiple model selections for consistency"""
    try:
        from core.decision_engine import EnhancedDecisionEngine
        
        engine = EnhancedDecisionEngine()
        
        test_cases = [
            "write code in python",
            "analyze business data", 
            "create a creative story",
            "help with strategy planning",
            "simple conversation"
        ]
        
        results = []
        for test_case in test_cases:
            start_time = time.time()
            model, confidence, reasoning = engine.select_model(test_case)
            end_time = time.time()
            
            results.append({
                "input": test_case,
                "model": model,
                "confidence": round(confidence, 3),
                "response_time": round((end_time - start_time) * 1000, 2),
                "pattern": reasoning.get("behavior_analysis", {}).get("detected_pattern", "unknown")
            })
        
        return True, results
        
    except Exception as e:
        return False, f"Error: {e}\n{traceback.format_exc()}"

# Streamlit UI
st.title("🧪 NAVA Testing Dashboard")
st.markdown("Interactive testing interface for NAVA Decision Engine")

# Sidebar for navigation
st.sidebar.title("Navigation")
test_mode = st.sidebar.selectbox(
    "Select Test Mode",
    ["Setup & Import", "Basic Functionality", "Feedback System", "Multiple Selections", "Debug Console"]
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if test_mode == "Setup & Import":
        st.header("🔧 Setup & Import Testing")
        
        if st.button("Test Path Setup"):
            with st.spinner("Setting up paths..."):
                path_result = setup_paths()
                
                if path_result:
                    st.success(f"✅ Path setup successful: {path_result}")
                else:
                    st.error("❌ Path setup failed")
        
        if st.button("Test Imports"):
            with st.spinner("Testing imports..."):
                import_success, import_message = test_imports()
                
                if import_success:
                    st.success(f"✅ Import successful: {import_message}")
                else:
                    st.error(f"❌ Import failed: {import_message}")
                    
                    # Show Python path for debugging
                    st.subheader("Python Path Debug Info")
                    for i, path in enumerate(sys.path):
                        st.text(f"{i}: {path}")
    
    elif test_mode == "Basic Functionality":
        st.header("⚙️ Basic Functionality Testing")
        
        if st.button("Test Basic Model Selection"):
            with st.spinner("Testing basic functionality..."):
                setup_paths()  # Ensure paths are set
                
                success, result = test_basic_functionality()
                
                if success:
                    st.success("✅ Basic functionality test passed!")
                    st.json(result)
                else:
                    st.error(f"❌ Basic functionality test failed: {result}")
        
        # Interactive model selection test
        st.subheader("Interactive Model Selection")
        user_input = st.text_input("Enter test message:", "write a business analysis")
        
        if st.button("Test Model Selection"):
            if user_input:
                setup_paths()
                try:
                    from core.decision_engine import EnhancedDecisionEngine
                    engine = EnhancedDecisionEngine()
                    
                    start_time = time.time()
                    model, confidence, reasoning = engine.select_model(user_input)
                    end_time = time.time()
                    
                    st.success("✅ Model selection successful!")
                    
                    # Display results
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Selected Model", model)
                    with col_b:
                        st.metric("Confidence", f"{confidence:.3f}")
                    with col_c:
                        st.metric("Response Time", f"{(end_time - start_time)*1000:.1f}ms")
                    
                    # Show reasoning
                    st.subheader("Decision Reasoning")
                    st.json(reasoning)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    elif test_mode == "Feedback System":
        st.header("📊 Feedback System Testing")
        
        if st.button("Test Feedback System"):
            with st.spinner("Testing feedback system..."):
                setup_paths()
                
                success, result = test_feedback_system()
                
                if success:
                    st.success("✅ Feedback system test passed!")
                    st.json(result)
                else:
                    st.error(f"❌ Feedback system test failed: {result}")
        
        # Interactive feedback submission
        st.subheader("Submit Feedback")
        feedback_col1, feedback_col2 = st.columns(2)
        
        with feedback_col1:
            response_id = st.text_input("Response ID", "test_feedback_001")
            model_used = st.selectbox("Model Used", ["gpt", "claude", "gemini"])
            pattern = st.selectbox("Pattern", ["conversation", "code_development", "deep_analysis", "strategic_planning"])
        
        with feedback_col2:
            feedback_score = st.slider("Feedback Score", 1.0, 5.0, 4.0, 0.1)
            feedback_type = st.selectbox("Feedback Type", ["rating", "thumbs", "regenerate"])
        
        if st.button("Submit Feedback"):
            setup_paths()
            try:
                from core.decision_engine import EnhancedDecisionEngine
                engine = EnhancedDecisionEngine()
                
                engine.update_user_feedback(response_id, model_used, pattern, feedback_score, feedback_type)
                
                st.success("✅ Feedback submitted successfully!")
                
                # Show updated stats
                stats = engine.get_feedback_stats()
                st.subheader("Updated Statistics")
                st.json(stats["feedback_summary"])
                
            except Exception as e:
                st.error(f"❌ Error submitting feedback: {e}")
    
    elif test_mode == "Multiple Selections":
        st.header("🔄 Multiple Selection Testing")
        
        if st.button("Run Multiple Selection Test"):
            with st.spinner("Running multiple selection tests..."):
                setup_paths()
                
                success, results = test_multiple_selections()
                
                if success:
                    st.success("✅ Multiple selection tests completed!")
                    
                    # Display results in a table
                    st.subheader("Test Results")
                    
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        avg_confidence = sum(r["confidence"] for r in results) / len(results)
                        st.metric("Average Confidence", f"{avg_confidence:.3f}")
                    
                    with col_b:
                        avg_response_time = sum(r["response_time"] for r in results) / len(results)
                        st.metric("Average Response Time", f"{avg_response_time:.1f}ms")
                    
                    with col_c:
                        unique_models = len(set(r["model"] for r in results))
                        st.metric("Models Used", f"{unique_models}/3")
                
                else:
                    st.error(f"❌ Multiple selection tests failed: {results}")
    
    elif test_mode == "Debug Console":
        st.header("🐛 Debug Console")
        
        st.subheader("Python Environment Info")
        st.text(f"Python Version: {sys.version}")
        st.text(f"Current Working Directory: {os.getcwd()}")
        
        st.subheader("Python Path")
        for i, path in enumerate(sys.path):
            st.text(f"{i}: {path}")
        
        st.subheader("File System Check")
        possible_paths = [
            "app/",
            "../app/",
            "./app/",
            "nava-logic-controller/app/",
            "../nava-logic-controller/app/"
        ]
        
        for path in possible_paths:
            path_obj = Path(path)
            if path_obj.exists():
                st.success(f"✅ Found: {path}")
                
                # List contents
                try:
                    contents = list(path_obj.iterdir())
                    st.text(f"   Contents: {[p.name for p in contents[:5]]}")
                except:
                    pass
            else:
                st.error(f"❌ Not found: {path}")
        
        # Custom Python code execution
        st.subheader("Custom Code Execution")
        custom_code = st.text_area(
            "Enter Python code to execute:",
            """# Example:
setup_paths()
from core.decision_engine import EnhancedDecisionEngine
engine = EnhancedDecisionEngine()
result = engine.select_model("test")
print(f"Result: {result}")
"""
        )
        
        if st.button("Execute Code"):
            try:
                # Create a safe execution environment
                exec_globals = {
                    'setup_paths': setup_paths,
                    'sys': sys,
                    'Path': Path,
                    'st': st
                }
                
                exec(custom_code, exec_globals)
                st.success("✅ Code executed successfully!")
                
            except Exception as e:
                st.error(f"❌ Execution error: {e}")
                st.text(traceback.format_exc())

with col2:
    st.header("📊 System Status")
    
    # Quick status check
    setup_paths()
    
    # Import status
    import_success, _ = test_imports()
    if import_success:
        st.success("✅ Imports Working")
    else:
        st.error("❌ Import Issues")
    
    # Basic functionality status
    try:
        basic_success, _ = test_basic_functionality()
        if basic_success:
            st.success("✅ Basic Functionality")
        else:
            st.error("❌ Basic Issues")
    except:
        st.error("❌ Basic Issues")
    
    # Current time
    st.text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**NAVA Testing Dashboard** - Interactive testing for decision engine")