#!/usr/bin/env python3
"""
Test NVIDIA API - Generate and Run Pandas Script for Zipcode Analysis

This script asks NVIDIA API to generate a pandas script for zipcode analysis,
then executes that generated script.
"""

import asyncio
import httpx
import json
import pandas as pd
import tempfile
import subprocess
import sys
import os

NVIDIA_API_KEY = "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"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"

# Sample NYC zipcode data (expanded dataset)
ZIPCODE_DATA = [
    {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5},
    {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2},
    {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8},
    {"zipcode": "10004", "population": 8000,  "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3},
    {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4},
    {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9},
    {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1},
    {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6},
    {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5},
    {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0}
]

async def generate_pandas_script():
    """Ask NVIDIA API to generate a pandas script for zipcode analysis"""
    print("🧪 Asking NVIDIA API to generate pandas script...")
    
    prompt = f"""
Create a complete Python pandas script that:
1. Creates a DataFrame from the provided zipcode data
2. Calculates incident rate per 1000 residents for each zipcode
3. Calculates incident density per square mile for each zipcode
4. Identifies high-risk zipcodes (above 75th percentile in incident rate)
5. Identifies low-risk zipcodes (below 25th percentile in incident rate)
6. Creates a summary table with zipcode, population, incident_rate_per_1000, incident_density_per_sqmi, and risk_level
7. Prints the full analysis results

Data to analyze: {json.dumps(ZIPCODE_DATA)}

Return ONLY the Python code, no explanations. Start with 'import pandas as pd'.
"""
    
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
                "max_tokens": 800
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            script_content = result["choices"][0]["message"]["content"]
            print("✅ NVIDIA API generated pandas script!")
            return script_content
        else:
            print(f"❌ NVIDIA API failed: {response.status_code} - {response.text}")
            return None

def clean_and_save_script(script_content):
    """Clean the generated script and save it to a file"""
    print("🧹 Cleaning and saving generated script...")
    
    # Remove any markdown code blocks
    if "```python" in script_content:
        script_content = script_content.split("```python")[1].split("```")[0]
    elif "```" in script_content:
        script_content = script_content.split("```")[1].split("```")[0]
    
    # Clean up the script
    script_content = script_content.strip()
    
    # Save to a temporary file
    script_path = "/workspaces/Innovate-tech-civic-ai-powered-research/test/generated_zipcode_analysis.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Script saved to: {script_path}")
    return script_path

def run_generated_script(script_path):
    """Execute the generated pandas script"""
    print("🚀 Running generated pandas script...")
    
    try:
        # Run the script and capture output
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Script executed successfully!")
            print("📊 ANALYSIS RESULTS:")
            print("=" * 60)
            print(result.stdout)
            return True
        else:
            print("❌ Script execution failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

async def test_nvidia_pandas_generation():
    """Main test function"""
    print("🚀 Testing NVIDIA API - Generate and Run Pandas Script")
    print("=" * 60)
    
    # Step 1: Generate the script
    script_content = await generate_pandas_script()
    if not script_content:
        print("❌ Failed to generate script")
        return False
    
    print("\n📝 Generated Script Preview:")
    print("-" * 40)
    print(script_content[:500] + "..." if len(script_content) > 500 else script_content)
    print("-" * 40)
    
    # Step 2: Clean and save the script
    script_path = clean_and_save_script(script_content)
    
    # Step 3: Run the script
    success = run_generated_script(script_path)
    
    if success:
        print("\n🎉 SUCCESS! NVIDIA API generated and executed a working pandas script!")
    else:
        print("\n⚠️  Script generation worked but execution had issues.")
    
    return success

if __name__ == "__main__":
    asyncio.run(test_nvidia_pandas_generation())