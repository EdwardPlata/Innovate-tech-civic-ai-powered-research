#!/usr/bin/env python3
"""
Run All Tests - Comprehensive AI Testing Suite

This script runs all AI functionality tests in sequence.
"""

import asyncio
import subprocess
import sys
import os

# Add current directory to path
sys.path.append('/workspaces/Innovate-tech-civic-ai-powered-research')

def run_script(script_name, description):
    """Run a test script and capture output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, 
            f"/workspaces/Innovate-tech-civic-ai-powered-research/test/{script_name}"
        ], capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Run all tests in sequence"""
    print("🧪 COMPREHENSIVE AI TESTING SUITE")
    print("Testing AI functionality across all components")
    
    tests = [
        ("test_api_keys.py", "Direct API Key Testing"),
        ("test_ai_functionality.py", "AI Functionality Package Testing"),
        ("test_backend_integration.py", "Backend API Integration Testing")
    ]
    
    results = {}
    
    for script, description in tests:
        results[description] = run_script(script, description)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL TEST RESULTS")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! AI functionality is working correctly.")
    elif passed > 0:
        print("⚠️  Some tests passed. AI functionality is partially working.")
    else:
        print("❌ All tests failed. AI functionality needs troubleshooting.")
    
    # Provide next steps
    print(f"\n{'='*60}")
    print("🔧 NEXT STEPS")
    print(f"{'='*60}")
    
    if results.get("Direct API Key Testing", False):
        print("✅ API keys are working")
    else:
        print("❌ Fix API keys first - check quotas and permissions")
    
    if results.get("AI Functionality Package Testing", False):
        print("✅ AI package is working")
    else:
        print("❌ Fix AI package imports and dependencies")
    
    if results.get("Backend API Integration Testing", False):
        print("✅ Backend integration is working")
    else:
        print("❌ Start backend server: cd backend && python run_server.py")

if __name__ == "__main__":
    main()