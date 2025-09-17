#!/usr/bin/env python3
"""
NVIDIA API Data Analysis Test
Test sending data for analysis like we would in the Scout system
"""

import asyncio
import httpx
import json

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

async def data_analysis_test():
    """Test NVIDIA API with data analysis request"""
    print("🚀 NVIDIA API Data Analysis Test")
    print("=" * 50)
    
    # Sample dataset (like what we'd send from Scout)
    sample_data = {
        "complaint_types": ["Noise", "Heating", "Plumbing", "Noise", "Heating", "Safety"],
        "boroughs": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Bronx", "Brooklyn"],
        "counts": [45, 32, 28, 67, 23, 41]
    }
    
    analysis_prompt = f"""
    Please analyze this NYC 311 complaints dataset and provide insights:
    
    Data: {json.dumps(sample_data, indent=2)}
    
    Please provide:
    1. A brief summary of the data
    2. Key trends or patterns you notice
    3. Which borough has the most complaints
    4. Most common complaint type
    5. Any recommendations for the city
    
    Keep your response concise and actionable.
    """
    
    print(f"📤 Sending data analysis request...")
    print(f"Data: {sample_data}")
    print()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                NVIDIA_BASE_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a data analyst specializing in NYC civic data analysis. Provide clear, actionable insights."
                        },
                        {
                            "role": "user", 
                            "content": analysis_prompt
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3
                },
                timeout=30.0
            )
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                ai_analysis = result["choices"][0]["message"]["content"]
                
                print("✅ Analysis Complete!")
                print("🔍 AI Analysis:")
                print("-" * 40)
                print(ai_analysis)
                print("-" * 40)
                
                print(f"\n📈 Token Usage:")
                usage = result.get("usage", {})
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
                
            else:
                print("❌ Failed!")
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(data_analysis_test())