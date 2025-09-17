#!/usr/bin/env python3
"""
Advanced NVIDIA API + Pandas Integration

This script demonstrates how to:
1. Ask NVIDIA API to generate pandas analysis code
2. Automatically execute the generated code
3. Handle real datasets with error correction
"""

import asyncio
import httpx
import json
import pandas as pd
import tempfile
import subprocess
import sys
import os
import re

NVIDIA_API_KEY = "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"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"

async def ask_nvidia_to_generate_analysis(dataset, analysis_request):
    """Ask NVIDIA API to generate specific pandas analysis code"""
    print(f"🧪 Asking NVIDIA API: {analysis_request}")
    
    prompt = f"""
Generate a complete Python pandas script that performs the following analysis:
{analysis_request}

Use this dataset: {json.dumps(dataset)}

Requirements:
1. Import pandas as pd
2. Create DataFrame from the provided data
3. Perform the requested analysis
4. Print clear, formatted results
5. Include summary statistics and insights
6. Return ONLY Python code, no markdown or explanations

The code should be complete and executable.
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
                "max_tokens": 1000
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            script_content = result["choices"][0]["message"]["content"]
            return script_content
        else:
            print(f"❌ NVIDIA API failed: {response.status_code} - {response.text}")
            return None

def clean_script(script_content):
    """Clean and prepare the generated script for execution"""
    # Remove markdown code blocks
    if "```python" in script_content:
        script_content = script_content.split("```python")[1].split("```")[0]
    elif "```" in script_content:
        script_content = script_content.split("```")[1].split("```")[0]
    
    # Clean up the script
    script_content = script_content.strip()
    
    # Ensure it starts with import
    if not script_content.startswith("import"):
        script_content = "import pandas as pd\n" + script_content
    
    return script_content

def execute_script(script_content, script_name):
    """Execute the generated pandas script"""
    script_path = f"/workspaces/Innovate-tech-civic-ai-powered-research/test/{script_name}.py"
    
    try:
        # Save script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Execute script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Script executed successfully!")
            print("📊 RESULTS:")
            print("-" * 50)
            print(result.stdout)
            return True
        else:
            print("❌ Script execution failed!")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def run_analysis_test(dataset, analysis_request, script_name):
    """Run a complete analysis test"""
    print(f"\n{'='*70}")
    print(f"🚀 ANALYSIS: {analysis_request}")
    print(f"{'='*70}")
    
    # Generate script
    script_content = await ask_nvidia_to_generate_analysis(dataset, analysis_request)
    if not script_content:
        return False
    
    # Clean and execute
    cleaned_script = clean_script(script_content)
    success = execute_script(cleaned_script, script_name)
    
    return success

async def main():
    """Run multiple analysis tests"""
    # Extended NYC dataset
    nyc_data = [
        {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5, "police_stations": 2},
        {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2, "police_stations": 3},
        {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8, "police_stations": 2},
        {"zipcode": "10004", "population": 8000,  "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3, "police_stations": 1},
        {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4, "police_stations": 1},
        {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9, "police_stations": 2},
        {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1, "police_stations": 3},
        {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6, "police_stations": 2},
        {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5, "police_stations": 4},
        {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0, "police_stations": 3}
    ]
    
    # Different analysis requests
    analyses = [
        ("Find the correlation between median income and incident rates. Create a scatter plot analysis.", "income_correlation"),
        ("Calculate police coverage efficiency (incidents per police station) for each zipcode.", "police_efficiency"),
        ("Identify the top 3 most dangerous zipcodes and recommend resource allocation.", "danger_ranking"),
        ("Analyze population density vs incident density to find patterns.", "density_analysis")
    ]
    
    print("🚀 NVIDIA API + PANDAS INTEGRATION TEST")
    print("Generating and executing multiple analysis scripts...")
    
    results = {}
    for analysis_request, script_name in analyses:
        success = await run_analysis_test(nyc_data, analysis_request, script_name)
        results[analysis_request] = success
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS")
    print(f"{'='*70}")
    
    for analysis, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {analysis[:50]}...")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\n🎯 Overall: {successful}/{total} analyses completed successfully")
    
    if successful > 0:
        print("🎉 NVIDIA API successfully generated and executed pandas analysis scripts!")
    else:
        print("⚠️  All analyses failed - check API connectivity and script generation")

if __name__ == "__main__":
    asyncio.run(main())