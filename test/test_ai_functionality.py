#!/usr/bin/env python3
"""
Test AI Functionality Integration

This script tests the AI_Functionality package integration with the backend.
"""

import sys
import asyncio
sys.path.append('/workspaces/Innovate-tech-civic-ai-powered-research')

try:
    from AI_Functionality import DataAnalyst, AnalysisType
    from AI_Functionality.core.ai_analyst import AIAnalyst
    print("✅ AI_Functionality imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_ai_analyst_creation():
    """Test creating AI analyst instance"""
    print("🧪 Testing AI Analyst Creation...")
    
    try:
        # Test with different providers
        providers_to_test = ["nvidia", "openai"]
        
        for provider in providers_to_test:
            print(f"  Testing with provider: {provider}")
            analyst = DataAnalyst(
                provider=provider,
                api_key="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" if provider == "nvidia" else "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"
            )
            print(f"  ✅ {provider} analyst created successfully")
            
        return True
        
    except Exception as e:
        print(f"❌ AI Analyst creation failed: {e}")
        return False

async def test_analysis_types():
    """Test AnalysisType enum"""
    print("🧪 Testing Analysis Types...")
    
    try:
        # Test all analysis types
        types = [
            AnalysisType.SUMMARY,
            AnalysisType.TRENDS,
            AnalysisType.INSIGHTS,
            AnalysisType.RECOMMENDATIONS
        ]
        
        for analysis_type in types:
            print(f"  ✅ {analysis_type.value} available")
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis types test failed: {e}")
        return False

async def test_simple_analysis():
    """Test simple data analysis"""
    print("🧪 Testing Simple Data Analysis...")
    
    try:
        # Create test data
        test_data = {
            "numbers": [1, 2, 3, 4, 5],
            "categories": ["A", "B", "A", "C", "B"],
            "description": "Simple test dataset with numbers and categories"
        }
        
        # Try with working provider (if any)
        providers_to_test = ["nvidia", "openai"]
        
        for provider in providers_to_test:
            try:
                print(f"  Testing analysis with {provider}...")
                
                analyst = DataAnalyst(
                    provider=provider,
                    api_key="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" if provider == "nvidia" else "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"
                )
                
                result = await analyst.analyze(
                    data=test_data,
                    analysis_type=AnalysisType.SUMMARY
                )
                
                if result and 'analysis' in result:
                    print(f"  ✅ {provider} analysis successful!")
                    print(f"    Result: {result['analysis'][:100]}...")
                    return True
                else:
                    print(f"  ❌ {provider} analysis returned empty result")
                    
            except Exception as e:
                print(f"  ❌ {provider} analysis failed: {e}")
                continue
                
        print("❌ All providers failed for analysis")
        return False
        
    except Exception as e:
        print(f"❌ Simple analysis test failed: {e}")
        return False

async def test_ai_functionality():
    """Run all AI functionality tests"""
    print("🚀 Testing AI Functionality Integration")
    print("=" * 50)
    
    tests = [
        ("AI Analyst Creation", test_ai_analyst_creation),
        ("Analysis Types", test_analysis_types),
        ("Simple Analysis", test_simple_analysis)
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
    
    if any(results.values()):
        print("\n🎉 Some AI functionality is working!")
    else:
        print("\n⚠️  All AI functionality tests failed.")

if __name__ == "__main__":
    asyncio.run(test_ai_functionality())