"""
Dataset Chat Service

A comprehensive chat interface for natural language dataset queries.
Integrates NVIDIA API with pandas analysis generation.
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
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

class DatasetChatService:
    def __init__(self, nvidia_api_key: str, memory_limit: int = 5):
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "meta/llama-3.1-8b-instruct"
        self.datasets = {}
        self.chat_history = []
        self.memory_limit = memory_limit
        
    def load_dataset(self, name: str, data: List[Dict]) -> bool:
        """Load a dataset for querying"""
        try:
            df = pd.DataFrame(data)
            self.datasets[name] = {
                'data': df,
                'raw_data': data,
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_types': df.dtypes.to_dict()
                }
            }
            return True
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
            return False
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict:
        """Get information about loaded datasets"""
        if dataset_name and dataset_name in self.datasets:
            return {
                'name': dataset_name,
                'info': self.datasets[dataset_name]['info'],
                'columns': self.datasets[dataset_name]['columns'],
                'sample_data': self.datasets[dataset_name]['data'].head(3).to_dict('records')
            }
        
        return {
            'available_datasets': list(self.datasets.keys()),
            'dataset_info': {name: info['info'] for name, info in self.datasets.items()}
        }
    
    async def generate_analysis_code(self, question: str, dataset_name: str) -> Optional[str]:
        """Generate pandas analysis code using NVIDIA API"""
        if dataset_name not in self.datasets:
            return None
            
        dataset_info = self.get_dataset_info(dataset_name)
        dataset_sample = dataset_info['sample_data']
        
        prompt = f"""
You are a data analyst assistant. Generate Python pandas code to answer the user's question.

DATASET: {dataset_name}
COLUMNS: {dataset_info['columns']}
SAMPLE DATA: {json.dumps(dataset_sample)}
TOTAL ROWS: {dataset_info['info']['rows']}

USER QUESTION: {question}

Generate complete Python code that:
1. Uses the variable 'df' (the dataset is already loaded)
2. Answers the user's question with analysis
3. Prints clear, formatted results
4. Includes relevant statistics and insights
5. Uses proper pandas operations

Return ONLY the Python code, no explanations or markdown.
Start with the analysis code (don't import pandas or load data).

Example format:
# Analysis for: {question}
result = df.groupby('column').sum()
print("Analysis Results:")
print(result)
print(f"\\nKey insight: ...")
"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.nvidia_api_url,
                    headers={
                        "Authorization": f"Bearer {self.nvidia_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 800
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    code = result["choices"][0]["message"]["content"]
                    return self._clean_generated_code(code)
                else:
                    print(f"NVIDIA API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"Error generating code: {e}")
            return None
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and prepare generated code for execution"""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Clean up the code
        code = code.strip()
        
        # Remove any imports or data loading (we'll handle this)
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if not (line.strip().startswith('import ') or 
                   line.strip().startswith('df = ') or
                   line.strip().startswith('data = ')):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def execute_analysis(self, code: str, dataset_name: str) -> Dict[str, Any]:
        """Execute analysis code on the dataset"""
        if dataset_name not in self.datasets:
            return {"error": "Dataset not found", "success": False}
        
        try:
            # Prepare execution environment
            df = self.datasets[dataset_name]['data'].copy()
            
            # Create a namespace for execution
            namespace = {
                'df': df,
                'pd': pd,
                'print': print,
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round
            }
            
            # Capture print output
            from io import StringIO
            import contextlib
            
            output_buffer = StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                exec(code, namespace)
            
            output = output_buffer.getvalue()
            
            return {
                "success": True,
                "output": output,
                "code": code,
                "dataset": dataset_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code": code,
                "dataset": dataset_name
            }
    
    async def chat(self, question: str, dataset_name: str = None) -> Dict[str, Any]:
        """Main chat interface for dataset queries"""
        timestamp = datetime.now().isoformat()
        
        # If no dataset specified, try to detect from available datasets
        if not dataset_name and self.datasets:
            dataset_name = list(self.datasets.keys())[0]
        
        if not dataset_name or dataset_name not in self.datasets:
            available = list(self.datasets.keys())
            return {
                "success": False,
                "error": f"No dataset specified or found. Available datasets: {available}",
                "timestamp": timestamp
            }
        
        # Add to chat history with memory management
        self._add_to_history({
            "type": "user_question",
            "question": question,
            "dataset": dataset_name,
            "timestamp": timestamp
        })
        
        # Generate analysis code
        code = await self.generate_analysis_code(question, dataset_name)
        if not code:
            return {
                "success": False,
                "error": "Failed to generate analysis code",
                "timestamp": timestamp
            }
        
        # Execute the analysis
        result = self.execute_analysis(code, dataset_name)
        
        # Add result to chat history with memory management
        self._add_to_history({
            "type": "ai_response",
            "question": question,
            "dataset": dataset_name,
            "result": result,
            "timestamp": timestamp
        })
        
        return {
            "success": result["success"],
            "question": question,
            "dataset": dataset_name,
            "code": code,
            "output": result.get("output", ""),
            "error": result.get("error"),
            "timestamp": timestamp
        }
    
    def get_chat_history(self) -> List[Dict]:
        """Get the complete chat history"""
        return self.chat_history

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []

    def set_memory_limit(self, limit: int):
        """Set the memory limit for chat history"""
        self.memory_limit = max(1, limit)  # Ensure at least 1 item
        self._trim_chat_history()

    def get_memory_limit(self) -> int:
        """Get the current memory limit"""
        return self.memory_limit

    def _trim_chat_history(self):
        """Trim chat history to memory limit, keeping most recent interactions"""
        if len(self.chat_history) > self.memory_limit:
            # Keep the most recent memory_limit items
            self.chat_history = self.chat_history[-self.memory_limit:]

    def _add_to_history(self, item: Dict):
        """Add item to history and manage memory limit"""
        self.chat_history.append(item)
        self._trim_chat_history()

# Example usage and testing
async def test_dataset_chat():
    """Test the dataset chat service"""
    
    # Initialize the service
    nvidia_key = "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"
    chat_service = DatasetChatService(nvidia_key)
    
    # Load sample NYC data
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
    
    # Load the dataset
    chat_service.load_dataset("nyc_zipcodes", nyc_data)
    
    # Test questions
    questions = [
        "Which zipcode has the highest crime rate?",
        "What's the correlation between income and incidents?",
        "Show me the top 3 safest neighborhoods",
        "Calculate the average income by police station count",
        "Which areas need more police coverage?"
    ]
    
    print("🚀 Testing Dataset Chat Service")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n💬 Question {i}: {question}")
        print("-" * 40)
        
        result = await chat_service.chat(question, "nyc_zipcodes")
        
        if result["success"]:
            print("✅ Analysis successful!")
            print("📊 Output:")
            print(result["output"])
        else:
            print("❌ Analysis failed!")
            print("Error:", result.get("error"))
    
    print(f"\n📝 Chat History: {len(chat_service.get_chat_history())} interactions")

if __name__ == "__main__":
    asyncio.run(test_dataset_chat())