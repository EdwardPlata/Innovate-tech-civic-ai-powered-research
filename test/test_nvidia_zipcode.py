#!/usr/bin/env python3
"""
Test NVIDIA API - Granular Zipcode Analysis

This script sends a granular analysis request to the NVIDIA API, searching per zipcode.
"""

import asyncio
import httpx
import json

NVIDIA_API_KEY = "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"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"

async def test_zipcode_analysis():
    print("🧪 Testing NVIDIA API - Zipcode Analysis...")
    # Example granular dataset
    dataset = [
        {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120},
        {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210},
        {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95},
        {"zipcode": "10004", "population": 8000,  "median_income": 90000, "incident_count": 15}
    ]
    prompt = (
        "Analyze the following NYC dataset by zipcode. "
        "For each zipcode, summarize the incident rate per 1000 residents, "
        "and highlight any zipcodes with unusually high or low rates. "
        "Return a concise table and a short summary.\n\n"
        f"DATASET: {json.dumps(dataset)}"
    )
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            NVIDIA_API_URL,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ NVIDIA API response:\n{content}")
        else:
            print(f"❌ NVIDIA API failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    asyncio.run(test_zipcode_analysis())
