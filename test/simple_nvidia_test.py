#!/usr/bin/env python3
"""
Simple NVIDIA API Test
Send a basic request and get a response
"""

import asyncio
import httpx
import json

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

async def simple_nvidia_test():
    """Send a simple test message to NVIDIA API"""
    print("🚀 Simple NVIDIA API Test")
    print("=" * 40)
    
    # Test message
    test_message = "Hello! Can you tell me what 2 + 2 equals? Just give me a short answer."
    
    print(f"📤 Sending message: {test_message}")
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
                            "role": "user", 
                            "content": test_message
                        }
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                
                print("✅ Success!")
                print(f"📥 AI Response: {ai_response}")
                print()
                print("📋 Full Response Structure:")
                print(json.dumps(result, indent=2))
                
            else:
                print("❌ Failed!")
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(simple_nvidia_test())