"""
Dataset Chat Component

Interactive chat interface for asking natural language questions about datasets.
Integrates with the backend chat service to provide AI-powered analysis.
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class DatasetChatComponent:
    def __init__(self, api_base_url: str = "http://localhost:8080/api"):
        self.api_base_url = api_base_url
        
    def load_dataset_to_chat(self, name: str, data: List[Dict]) -> bool:
        """Load a dataset into the chat service"""
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/load-dataset",
                json={"name": name, "data": data},
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return False
    
    def ask_question(self, question: str, dataset_name: str = None) -> Dict:
        """Ask a question about a dataset"""
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/ask",
                json={"question": question, "dataset_name": dataset_name},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_loaded_datasets(self) -> Dict:
        """Get information about loaded datasets"""
        try:
            response = requests.get(f"{self.api_base_url}/chat/datasets")
            if response.status_code == 200:
                return response.json()
            else:
                return {"available_datasets": [], "dataset_info": {}}
        except Exception as e:
            st.error(f"Failed to get datasets: {e}")
            return {"available_datasets": [], "dataset_info": {}}
    
    def get_chat_history(self) -> List[Dict]:
        """Get the chat history"""
        try:
            response = requests.get(f"{self.api_base_url}/chat/history")
            if response.status_code == 200:
                return response.json().get("chat_history", [])
            else:
                return []
        except Exception as e:
            st.error(f"Failed to get chat history: {e}")
            return []
    
    def clear_chat_history(self) -> bool:
        """Clear the chat history"""
        try:
            response = requests.post(f"{self.api_base_url}/chat/clear-history")
            return response.status_code == 200
        except Exception as e:
            st.error(f"Failed to clear history: {e}")
            return False

    def get_memory_config(self) -> Dict:
        """Get current memory configuration"""
        try:
            response = requests.get(f"{self.api_base_url}/chat/memory-config")
            if response.status_code == 200:
                return response.json()
            else:
                return {"memory_limit": 5, "current_history_length": 0, "success": False, "message": "Failed to get config"}
        except Exception as e:
            st.error(f"Failed to get memory config: {e}")
            return {"memory_limit": 5, "current_history_length": 0, "success": False, "message": str(e)}

    def set_memory_config(self, memory_limit: int) -> Dict:
        """Set memory configuration"""
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/memory-config",
                json={"memory_limit": memory_limit},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}
        except Exception as e:
            st.error(f"Failed to set memory config: {e}")
            return {"success": False, "message": str(e)}
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🤖 Dataset Chat Assistant")
        st.markdown("Ask natural language questions about your datasets and get AI-powered analysis!")
        
        # Initialize session state
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        
        # Sidebar for dataset management
        with st.sidebar:
            self._render_dataset_sidebar()
        
        # Main chat interface
        self._render_chat_area()
    
    def _render_dataset_sidebar(self):
        """Render the dataset management sidebar"""
        st.header("📊 Dataset Management")
        
        # Load dataset section
        with st.expander("📥 Load New Dataset", expanded=False):
            dataset_name = st.text_input("Dataset Name", placeholder="e.g., nyc_311_calls")
            
            # Option to upload CSV or use sample data
            data_source = st.radio("Data Source", ["Upload CSV", "Use Sample Data"])
            
            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                if uploaded_file is not None and dataset_name:
                    try:
                        df = pd.read_csv(uploaded_file)
                        # Limit to first 1000 rows for demo
                        if len(df) > 1000:
                            df = df.head(1000)
                            st.warning("Dataset limited to first 1000 rows for this demo")
                        
                        data = df.to_dict('records')
                        
                        if st.button("Load Dataset"):
                            with st.spinner("Loading dataset..."):
                                success = self.load_dataset_to_chat(dataset_name, data)
                                if success:
                                    st.success(f"Dataset '{dataset_name}' loaded successfully!")
                                    st.session_state.current_dataset = dataset_name
                                    st.rerun()
                                else:
                                    st.error("Failed to load dataset")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
            
            elif data_source == "Use Sample Data":
                sample_datasets = {
                    "NYC Zipcodes Sample": [
                        {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5, "police_stations": 2},
                        {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2, "police_stations": 3},
                        {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8, "police_stations": 2},
                        {"zipcode": "10004", "population": 8000, "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3, "police_stations": 1},
                        {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4, "police_stations": 1},
                        {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9, "police_stations": 2},
                        {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1, "police_stations": 3}
                    ]
                }
                
                selected_sample = st.selectbox("Choose Sample Dataset", list(sample_datasets.keys()))
                
                if st.button("Load Sample Dataset") and dataset_name:
                    with st.spinner("Loading sample dataset..."):
                        data = sample_datasets[selected_sample]
                        success = self.load_dataset_to_chat(dataset_name, data)
                        if success:
                            st.success(f"Sample dataset '{dataset_name}' loaded successfully!")
                            st.session_state.current_dataset = dataset_name
                            st.rerun()
                        else:
                            st.error("Failed to load sample dataset")
        
        # Current datasets section
        st.subheader("📋 Loaded Datasets")
        datasets_info = self.get_loaded_datasets()
        available_datasets = datasets_info.get("available_datasets", [])
        
        if available_datasets:
            # Dataset selector
            selected_dataset = st.selectbox(
                "Select Active Dataset",
                available_datasets,
                index=available_datasets.index(st.session_state.current_dataset) if st.session_state.current_dataset in available_datasets else 0
            )
            st.session_state.current_dataset = selected_dataset
            
            # Dataset info
            dataset_info = datasets_info.get("dataset_info", {}).get(selected_dataset, {})
            if dataset_info:
                st.write(f"**Rows:** {dataset_info.get('rows', 'Unknown')}")
                st.write(f"**Columns:** {dataset_info.get('columns', 'Unknown')}")
        else:
            st.info("No datasets loaded. Load a dataset to start chatting!")
        
        # Chat management
        st.subheader("🗂️ Chat Management")

        # Memory configuration
        with st.expander("🧠 Memory Settings", expanded=False):
            # Get current memory config
            current_config = self.get_memory_config()
            current_limit = current_config.get("memory_limit", 5)
            current_history_length = current_config.get("current_history_length", 0)

            st.write(f"**Current Memory Limit:** {current_limit} items")
            st.write(f"**Current History Length:** {current_history_length} items")

            # Memory limit selector
            new_memory_limit = st.slider(
                "Chat Memory Limit",
                min_value=1,
                max_value=50,
                value=current_limit,
                help="Number of chat history items to remember (default: 5)"
            )

            # Update button
            if st.button("Update Memory Limit"):
                with st.spinner("Updating memory configuration..."):
                    result = self.set_memory_config(new_memory_limit)
                    if result.get("success"):
                        st.success(f"Memory limit updated to {new_memory_limit} items!")
                        if result.get("current_history_length", 0) < current_history_length:
                            st.info(f"Chat history trimmed to {result.get('current_history_length', 0)} items")
                        st.rerun()
                    else:
                        st.error(f"Failed to update memory limit: {result.get('message', 'Unknown error')}")

            st.markdown("---")

        # Clear history button
        if st.button("Clear Chat History"):
            if self.clear_chat_history():
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")
                st.rerun()
    
    def _render_chat_area(self):
        """Render the main chat area"""
        # Check if we have an active dataset
        if not st.session_state.current_dataset:
            st.info("👈 Please load and select a dataset from the sidebar to start chatting!")
            return
        
        # Display current dataset info and memory status
        memory_config = self.get_memory_config()
        col1, col2 = st.columns([3, 1])

        with col1:
            st.info(f"💬 Chatting about: **{st.session_state.current_dataset}**")

        with col2:
            memory_limit = memory_config.get("memory_limit", 5)
            history_length = memory_config.get("current_history_length", 0)
            st.metric(
                "🧠 Memory",
                f"{history_length}/{memory_limit}",
                help=f"Current chat history items: {history_length} out of {memory_limit} limit"
            )
        
        # Chat container
        chat_container = st.container()
        
        # Question input
        col1, col2 = st.columns([6, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question about your dataset:",
                placeholder="e.g., Which zipcode has the highest crime rate?",
                key="question_input"
            )
        
        with col2:
            ask_button = st.button("Ask", type="primary")
        
        # Process question
        if (ask_button or question) and question.strip():
            self._process_question(question.strip(), chat_container)
        
        # Display chat history
        with chat_container:
            self._display_chat_history()
        
        # Suggested questions
        self._render_suggested_questions()
    
    def _process_question(self, question: str, container):
        """Process a user question"""
        with container:
            # Add user message to session state
            st.session_state.chat_messages.append({
                "type": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            
            # Show user message
            with st.chat_message("user"):
                st.write(question)
            
            # Show AI thinking
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your question..."):
                    result = self.ask_question(question, st.session_state.current_dataset)
                
                if result.get("success"):
                    # Display the analysis output
                    st.write("📊 **Analysis Results:**")
                    
                    if result.get("output"):
                        st.code(result["output"], language="text")
                    
                    # Show generated code in an expander
                    if result.get("code"):
                        with st.expander("🔍 View Generated Code"):
                            st.code(result["code"], language="python")
                    
                    # Add to session state
                    st.session_state.chat_messages.append({
                        "type": "assistant",
                        "content": result.get("output", "Analysis completed"),
                        "code": result.get("code"),
                        "timestamp": result.get("timestamp", datetime.now().isoformat())
                    })
                else:
                    error_msg = result.get("error", "Unknown error occurred")
                    st.error(f"❌ Analysis failed: {error_msg}")
                    
                    st.session_state.chat_messages.append({
                        "type": "assistant",
                        "content": f"Error: {error_msg}",
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Clear the input
        st.session_state.question_input = ""
    
    def _display_chat_history(self):
        """Display the chat history"""
        for message in st.session_state.chat_messages:
            if message["type"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show code if available
                    if message.get("code"):
                        with st.expander("🔍 View Code"):
                            st.code(message["code"], language="python")
    
    def _render_suggested_questions(self):
        """Render suggested questions based on the current dataset"""
        if not st.session_state.current_dataset:
            return
        
        st.subheader("💡 Suggested Questions")
        
        # Generic suggestions that work for most datasets
        suggestions = [
            "What are the summary statistics for this dataset?",
            "Which column has the most variation?",
            "Show me the top 5 records by value",
            "Are there any missing values in the dataset?",
            "What are the correlations between numeric columns?",
            "Create a simple visualization of the data",
            "What insights can you find in this dataset?"
        ]
        
        # Dataset-specific suggestions
        if "zipcode" in st.session_state.current_dataset.lower():
            suggestions.extend([
                "Which zipcode has the highest incident rate?",
                "What's the correlation between income and incidents?",
                "Show me the safest neighborhoods",
                "Which areas need more police coverage?"
            ])
        
        # Display suggestions as buttons
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:8]):  # Limit to 8 suggestions
            col = cols[i % 2]
            if col.button(suggestion, key=f"suggestion_{i}"):
                # Set the question and process it
                st.session_state.question_input = suggestion
                self._process_question(suggestion, st.container())

def get_dataset_chat_component() -> DatasetChatComponent:
    """Get or create the dataset chat component"""
    if 'chat_component' not in st.session_state:
        st.session_state.chat_component = DatasetChatComponent()
    return st.session_state.chat_component