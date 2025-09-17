#!/usr/bin/env python3
"""
Test AI_Functionality with NVIDIA API
Test the actual Scout AI integration
"""

import sys
import asyncio
sys.path.append('/workspaces/Innovate-tech-civic-ai-powered-research')

try:
    from AI_Functionality import DataAnalyst, AnalysisType
    print("✅ AI_Functionality imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_scout_ai_integration():
    """Test Scout AI integration with NVIDIA"""
    print("🚀 Testing Scout AI Integration with NVIDIA")
    print("=" * 50)
    
    # Test data similar to what Scout would process
    nyc_data = {
        "dataset_name": "NYC 311 Complaints Sample",
        "complaint_types": ["Noise", "Heating", "Plumbing", "Noise", "Heating", "Safety"],
        "boroughs": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Bronx", "Brooklyn"],
        "counts": [45, 32, 28, 67, 23, 41],
        "description": "Sample NYC 311 complaints data for testing AI analysis functionality"
    }
    
    try:
        # Create DataAnalyst with NVIDIA provider
        print("📝 Creating DataAnalyst with NVIDIA provider...")
        analyst = DataAnalyst(
            provider="nvidia",
            api_key="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"
        )
        print("✅ DataAnalyst created successfully")
        
        # Test different analysis types
        analysis_types = [
            AnalysisType.SUMMARY,
            AnalysisType.INSIGHTS,
            AnalysisType.TRENDS
        ]
        
        for analysis_type in analysis_types:
            print(f"\n🔍 Testing {analysis_type.value} analysis...")
            
            try:
                result = await analyst.analyze(
                    data=nyc_data,
                    analysis_type=analysis_type
                )
                
                if result and 'analysis' in result:
                    print(f"✅ {analysis_type.value} analysis successful!")
                    print(f"📄 Result preview: {result['analysis'][:200]}...")
                    if 'metadata' in result:
                        print(f"📊 Metadata: {result['metadata']}")
                else:
                    print(f"❌ {analysis_type.value} analysis returned empty result")
                    
            except Exception as e:
                print(f"❌ {analysis_type.value} analysis failed: {e}")
        
        print(f"\n🎉 Scout AI integration test complete!")
        
    except Exception as e:
        print(f"❌ Scout AI integration test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_scout_ai_integration())