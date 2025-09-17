#!/usr/bin/env python3
"""
Test Backend API Integration

This script tests the Scout backend API endpoints for AI functionality.
"""

import asyncio
import httpx
import json

BACKEND_URL = "http://localhost:8080"

async def test_backend_health():
    """Test if backend is running"""
    print("🧪 Testing Backend Health...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=10.0)
            
            if response.status_code == 200:
                print("✅ Backend is running and healthy")
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        print("💡 Make sure backend is running with: cd backend && python run_server.py")
        return False

async def test_ai_config_endpoint():
    """Test AI configuration endpoint"""
    print("🧪 Testing AI Configuration Endpoint...")
    
    try:
        config_data = {
            "provider": "nvidia",
            "api_key": "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"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/ai/config",
                json=config_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ AI config successful: {result}")
                return True
            else:
                print(f"❌ AI config failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ AI config endpoint error: {e}")
        return False

async def test_ai_analyze_endpoint():
    """Test AI analysis endpoint"""
    print("🧪 Testing AI Analysis Endpoint...")
    
    try:
        # First configure AI
        config_data = {
            "provider": "nvidia",
            "api_key": "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"
        }
        
        analysis_data = {
            "data": {
                "test_data": [1, 2, 3, 4, 5],
                "description": "Simple test dataset for API testing"
            },
            "analysis_type": "summary"
        }
        
        async with httpx.AsyncClient() as client:
            # Configure AI first
            config_response = await client.post(
                f"{BACKEND_URL}/api/ai/config",
                json=config_data,
                timeout=30.0
            )
            
            if config_response.status_code != 200:
                print(f"❌ AI config failed, can't test analysis")
                return False
            
            # Now test analysis
            analyze_response = await client.post(
                f"{BACKEND_URL}/api/ai/analyze",
                json=analysis_data,
                timeout=60.0  # Analysis might take longer
            )
            
            if analyze_response.status_code == 200:
                result = analyze_response.json()
                print(f"✅ AI analysis successful!")
                if 'analysis' in result:
                    print(f"    Analysis result: {result['analysis'][:100]}...")
                return True
            else:
                print(f"❌ AI analysis failed: {analyze_response.status_code} - {analyze_response.text}")
                return False
                
    except Exception as e:
        print(f"❌ AI analysis endpoint error: {e}")
        return False

async def test_datasets_endpoint():
    """Test datasets endpoint"""
    print("🧪 Testing Datasets Endpoint...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/datasets", timeout=10.0)
            
            if response.status_code == 200:
                datasets = response.json()
                print(f"✅ Datasets endpoint works! Found {len(datasets)} datasets")
                if datasets:
                    print(f"    First dataset: {list(datasets.keys())[0]}")
                return True
            else:
                print(f"❌ Datasets endpoint failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Datasets endpoint error: {e}")
        return False

async def test_backend_integration():
    """Run all backend integration tests"""
    print("🚀 Testing Backend API Integration")
    print("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Datasets Endpoint", test_datasets_endpoint),
        ("AI Config Endpoint", test_ai_config_endpoint),
        ("AI Analysis Endpoint", test_ai_analyze_endpoint)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    if results.get("Backend Health", False):
        if any(results.values()):
            print("\n🎉 Backend is working with some AI functionality!")
        else:
            print("\n⚠️  Backend is running but AI functionality has issues.")
    else:
        print("\n⚠️  Backend is not running. Start it with: cd backend && python run_server.py")

if __name__ == "__main__":
    asyncio.run(test_backend_integration())