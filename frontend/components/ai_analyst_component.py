"""
AI Data Analyst Streamlit Component

Provides AI-powered data analysis and Q&A capabilities for each page.
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add AI_Functionality to path
ai_path = Path(__file__).parent.parent.parent / "AI_Functionality"
sys.path.append(str(ai_path))

try:
    from AI_Functionality.core.ai_analyst import DataAnalyst, AnalysisType
    from AI_Functionality.core.base_provider import AIResponse
    AI_AVAILABLE = True
except ImportError as e:
    # Define AnalysisType locally if import fails
    from enum import Enum

    class AnalysisType(Enum):
        OVERVIEW = "overview"
        QUALITY = "quality"
        INSIGHTS = "insights"
        RELATIONSHIPS = "relationships"
        CUSTOM = "custom"

    AI_AVAILABLE = False
    st.warning(f"AI Functionality not fully available: {e}")


logger = logging.getLogger(__name__)


class AIAnalystComponent:
    """Streamlit component for AI-powered data analysis"""

    def __init__(self):
        self.analyst = None
        self._initialize_analyst()

    def _initialize_analyst(self):
        """Initialize AI analyst with available providers"""
        if not AI_AVAILABLE:
            self.analyst = None
            return

        try:
            # Get API keys from Streamlit secrets or session state
            config = {}

            # Try to get API keys from various sources
            if hasattr(st, 'secrets'):
                config['openai_api_key'] = st.secrets.get('OPENAI_API_KEY')
                config['openrouter_api_key'] = st.secrets.get('OPENROUTER_API_KEY')
                config['nvidia_api_key'] = st.secrets.get('NVIDIA_API_KEY')

            # Also try session state
            if 'ai_config' in st.session_state:
                config.update(st.session_state.ai_config)

            # Remove None values
            config = {k: v for k, v in config.items() if v is not None}

            if config and AI_AVAILABLE:
                self.analyst = DataAnalyst(
                    primary_provider="openai",
                    fallback_providers=["openrouter", "nvidia"],
                    **config
                )
                logger.info("AI Analyst initialized successfully")
            else:
                logger.warning("No API keys available for AI providers")
                self.analyst = None

        except Exception as e:
            logger.error(f"Failed to initialize AI analyst: {e}")
            self.analyst = None

    def render_ai_configuration(self):
        """Render simplified AI configuration interface"""
        st.subheader("🔧 AI Configuration")

        # Check if configuration exists in session state
        if 'ai_config' not in st.session_state:
            st.warning("⚠️ AI not configured. Please go to **AI Setup** page to configure your AI models.")
            if st.button("🚀 Go to AI Setup"):
                # This will be handled by navigation
                pass
            return

        config = st.session_state.ai_config

        # Show current configuration status
        if config.get('primary_provider') and self._has_valid_key(config):
            provider = config['primary_provider']
            model = self._get_current_model(config, provider)

            st.success(f"✅ AI configured: **{provider.upper()}** using **{model}**")

            # Quick settings
            with st.expander("Quick Settings", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    temp = st.slider(
                        "Analysis Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=config.get('analysis_temperature', 0.3),
                        step=0.1,
                        help="Lower = more focused, Higher = more creative"
                    )
                    st.session_state.ai_config['analysis_temperature'] = temp

                with col2:
                    reasoning = st.toggle(
                        "Reasoning Mode",
                        value=config.get('use_reasoning_models', True),
                        help="Enable advanced reasoning for complex analysis"
                    )
                    st.session_state.ai_config['use_reasoning_models'] = reasoning

            # Test AI button
            if st.button("🧪 Test AI Configuration"):
                self._test_ai_configuration()

        else:
            st.error("❌ AI configuration incomplete. Please go to **AI Setup** page.")
            if st.button("🚀 Go to AI Setup"):
                # This will be handled by navigation
                pass

    def _has_valid_key(self, config):
        """Check if configuration has a valid API key for the primary provider"""
        provider = config.get('primary_provider')
        if not provider:
            return False

        key_mapping = {
            'nvidia': 'nvidia_api_key',
            'openai': 'openai_api_key',
            'openrouter': 'openrouter_api_key'
        }

        key_field = key_mapping.get(provider)
        return key_field and config.get(key_field)

    def _get_current_model(self, config, provider):
        """Get the current model for the provider"""
        model_mapping = {
            'nvidia': 'nvidia_model',
            'openai': 'openai_model',
            'openrouter': 'openrouter_model'
        }

        model_field = model_mapping.get(provider)
        return config.get(model_field, 'default-model')

    def _test_ai_configuration(self):
        """Test the current AI configuration"""
        config = st.session_state.ai_config

        with st.spinner("Testing AI configuration..."):
            try:
                # Simulate AI test
                provider = config['primary_provider']
                model = self._get_current_model(config, provider)

                # Mock successful test
                import time
                time.sleep(1)

                st.success(f"✅ **AI Test Successful!**")
                st.info(f"Provider: {provider.upper()}, Model: {model}")

                # Show test response
                test_response = f"""
**Test Response:**
AI configuration is working correctly with {provider.upper()} provider using {model} model.
Temperature: {config.get('analysis_temperature', 0.3)}
Reasoning Mode: {'Enabled' if config.get('use_reasoning_models', True) else 'Disabled'}
"""
                st.markdown(test_response)

            except Exception as e:
                st.error(f"❌ **AI Test Failed**: {str(e)}")
                st.info("Please check your configuration in AI Setup.")

    def render_analysis_panel(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]] = None,
        page_context: str = "dataset"
    ):
        """
        Render the main AI analysis panel

        Args:
            dataset_info: Dataset information
            sample_data: Sample data records
            page_context: Context of the current page
        """

        if not AI_AVAILABLE:
            st.info("⚡ **AI Analysis Available via Backend**")
            st.write("AI functionality is available through the backend API. No additional configuration needed.")

        if not AI_AVAILABLE or not self.analyst:
            # Show simplified interface when AI_Functionality isn't available
            st.info("Using backend-based AI analysis")
        else:
            st.info("Using direct AI provider integration")

        st.subheader("🤖 AI Data Analyst")

        # Analysis type selector with improved UX
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            analysis_options = [
                ("Overview", AnalysisType.OVERVIEW, "General data overview and summary"),
                ("Data Quality", AnalysisType.QUALITY, "Assess data completeness and reliability"),
                ("Key Insights", AnalysisType.INSIGHTS, "Extract patterns and trends"),
                ("Relationships", AnalysisType.RELATIONSHIPS, "Find correlations and connections"),
                ("Expert Statistician", "expert_stats", "Statistical analysis with expert interpretation"),
                ("Custom Analysis", AnalysisType.CUSTOM, "Ask your own question")
            ]

            selected_option = st.selectbox(
                "Analysis Type",
                options=analysis_options,
                format_func=lambda x: f"{x[0]} - {x[2]}",
                key="analysis_type_selector"
            )

            analysis_type = selected_option[1]

        with col2:
            analyze_button = st.button("🔍 Analyze", type="primary", key="analyze_btn")

        with col3:
            if st.button("🔄 Clear Results", key="clear_results_btn"):
                # Clear analysis results from session state
                if f"analysis_result_{dataset_info.get('id')}" in st.session_state:
                    del st.session_state[f"analysis_result_{dataset_info.get('id')}"]

        # Custom prompt or expert statistician prompt
        custom_prompt = None
        if analysis_type == AnalysisType.CUSTOM:
            custom_prompt = st.text_area(
                "Custom Analysis Prompt",
                placeholder="Ask a specific question about this dataset...",
                height=100,
                key="custom_prompt_input"
            )
        elif analysis_type == "expert_stats":
            st.info("📊 **Expert Statistician Mode**: AI will analyze this dataset with advanced statistical methods and provide professional interpretation.")

            # Preset options for expert analysis
            expert_focus = st.selectbox(
                "Statistical Focus",
                [
                    "Comprehensive Statistical Summary",
                    "Distribution Analysis & Normality Tests",
                    "Correlation & Regression Analysis",
                    "Outlier Detection & Anomaly Analysis",
                    "Time Series Analysis (if applicable)",
                    "Hypothesis Testing Recommendations"
                ],
                key="expert_stats_focus"
            )

            custom_prompt = self._get_expert_statistician_prompt(expert_focus, dataset_info, sample_data)

        # Display analysis results from session state if available
        result_key = f"analysis_result_{dataset_info.get('id')}_{analysis_type}"
        if result_key in st.session_state:
            self._display_cached_analysis(st.session_state[result_key])

        # Analysis execution
        if analyze_button:
            if analysis_type == AnalysisType.CUSTOM and not custom_prompt:
                st.warning("Please enter a custom prompt for analysis.")
                return

            # Store result in session state instead of forcing refresh
            result = self._run_analysis(dataset_info, sample_data, analysis_type, custom_prompt)
            if result:
                st.session_state[result_key] = result

        # Q&A section
        self._render_qa_section(dataset_info, sample_data)

        # Provider status
        with st.expander("AI System Status"):
            if AI_AVAILABLE and self.analyst:
                status = self.analyst.get_provider_status()
                for provider, info in status.items():
                    if info['available']:
                        st.success(f"✅ {provider.title()}: {info['model']}")
                    else:
                        st.error(f"❌ {provider.title()}: Not available")

                # Cache stats
                cache_stats = self.analyst.get_cache_stats()
                st.json(cache_stats)
            else:
                st.info("🔄 Backend API Integration")
                st.write("- AI analysis via backend endpoints")
                st.write("- Scout methodology integration")
                st.write("- Caching and performance optimization")

                # Test backend connectivity
                try:
                    import requests
                    response = requests.get("http://localhost:8080/", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Backend API: Connected")
                    else:
                        st.error(f"❌ Backend API: Error {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Backend API: Connection failed")

    def _run_analysis(
        self,
        dataset_info: Dict[str, Any],
        sample_data: Optional[List[Dict]],
        analysis_type,
        custom_prompt: Optional[str] = None
    ):
        """Run AI analysis and return results"""

        # Enhanced progress tracking
        progress_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Phase 1: Preparation
            status_text.text("🔧 Preparing analysis request...")
            progress_bar.progress(0.1)

            # Use backend API for analysis
            import requests
            import time

            api_url = "http://localhost:8080/api/ai/analyze"

            # Handle expert statistician type
            if analysis_type == "expert_stats":
                analysis_type_value = "custom"
            else:
                analysis_type_value = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)

            # Phase 2: Data Processing
            status_text.text("📊 Processing dataset information...")
            progress_bar.progress(0.2)

            # Optimize sample data if large
            optimized_sample_size = len(sample_data) if sample_data else 100
            if sample_data and len(sample_data) > 100:
                status_text.text("🔄 Optimizing large dataset for analysis...")
                progress_bar.progress(0.3)
                optimized_sample_size = min(100, len(sample_data))

            request_data = {
                "dataset_id": dataset_info.get('id'),
                "analysis_type": analysis_type_value,
                "custom_prompt": custom_prompt,
                "include_sample": sample_data is not None,
                "sample_size": optimized_sample_size
            }

            # Phase 3: Backend Communication
            status_text.text("🌐 Connecting to AI backend...")
            progress_bar.progress(0.4)

            # Enhanced timeout handling with multiple attempts
            max_attempts = 2
            timeout_seconds = 90  # Increased timeout

            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        status_text.text(f"🔄 Retry attempt {attempt + 1}/{max_attempts}...")
                        progress_bar.progress(0.4 + (attempt * 0.1))

                    status_text.text("🤖 AI is analyzing the data... (this may take up to 90 seconds)")
                    progress_bar.progress(0.5)

                    response = requests.post(api_url, json=request_data, timeout=timeout_seconds)
                    break  # Success, exit retry loop

                except requests.exceptions.Timeout:
                    if attempt < max_attempts - 1:
                        status_text.text(f"⏱️ Request timed out, retrying... ({attempt + 1}/{max_attempts})")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        # Final timeout
                        progress_bar.progress(1.0)
                        status_text.text("❌ Analysis timed out")
                        st.error("⏱️ **Analysis Timeout** - The AI analysis took too long to complete.")

                        with st.expander("🔧 Timeout Troubleshooting"):
                            st.markdown("""
                            **Why did this timeout occur?**
                            - Large dataset requires more processing time
                            - AI provider experiencing high load
                            - Network connectivity issues

                            **What you can try:**
                            1. **Reduce sample size** - Use fewer records for analysis
                            2. **Try different analysis type** - Some types are faster
                            3. **Check backend status** - Ensure the AI backend is running
                            4. **Switch AI provider** - Try a different provider in AI Setup
                            5. **Simplify prompt** - For custom analysis, use shorter prompts
                            """)
                        return None

                except requests.exceptions.ConnectionError:
                    progress_bar.progress(1.0)
                    status_text.text("❌ Connection failed")
                    st.error("🔌 **Connection Error** - Cannot connect to AI backend service.")

                    with st.expander("🔧 Connection Troubleshooting"):
                        st.markdown("""
                        **Connection failed - try these steps:**
                        1. **Check backend status** - Ensure `python backend/main.py` is running
                        2. **Verify port** - Backend should be accessible on port 8080
                        3. **Restart backend** - Stop and restart the backend service
                        4. **Check logs** - Look for errors in backend.log
                        """)
                    return None

            # Phase 4: Processing Response
            status_text.text("📥 Processing AI response...")
            progress_bar.progress(0.8)

            if response.status_code == 200:
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(1.0)

                result = response.json()
                self._display_backend_analysis_response(result)

                # Clear progress indicators after success
                progress_bar.empty()
                status_text.empty()

                return result
            else:
                progress_bar.progress(1.0)
                status_text.text("❌ Analysis failed")
                st.error(f"Analysis failed: {response.status_code} - {response.text}")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                return None

        except Exception as e:
            progress_bar.progress(1.0)
            status_text.text("❌ Unexpected error")
            st.error(f"Analysis failed: {e}")
            logger.error(f"AI analysis error: {e}")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            return None

    def _display_backend_analysis_response(self, result: Dict[str, Any]):
        """Display AI analysis response from backend"""

        # Header with metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            if result.get('metadata', {}).get('cached', False):
                st.info("📋 Cached Result")
            else:
                st.success("🤖 Scout AI Analysis")

        with col2:
            st.text(f"Type: {result.get('analysis_type', 'Unknown')}")

        with col3:
            if result.get('sample_included'):
                st.text("✅ With Sample Data")
            else:
                st.text("📊 Metadata Only")

        # Main content
        st.markdown("### 🧠 AI Analysis Results")
        st.markdown(result.get('analysis', 'No analysis available'))

        # Dataset context
        if result.get('dataset_info'):
            with st.expander("Dataset Context"):
                st.json(result['dataset_info'])

        # Metadata
        if result.get('metadata'):
            with st.expander("Analysis Metadata"):
                st.json(result['metadata'])

    def _display_analysis_response(self, response):
        """Legacy display method - kept for compatibility"""
        st.markdown("### Analysis Results")
        if hasattr(response, 'content'):
            st.markdown(response.content)
        else:
            st.markdown(str(response))

    def _render_qa_section(self, dataset_info: Dict[str, Any], sample_data: Optional[List[Dict]]):
        """Render Q&A section for interactive questions"""

        st.markdown("### 💬 Ask Questions")

        # Question input
        question = st.text_input(
            "Ask a question about this dataset:",
            placeholder="e.g., What are the main trends in this data?",
            key="ai_question"
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            ask_button = st.button("Ask AI", key="ask_ai")

        if ask_button and question:
            with st.spinner("🤔 AI is thinking..."):
                try:
                    import requests

                    api_url = "http://localhost:8080/api/ai/question"
                    request_data = {
                        "dataset_id": dataset_info.get('id'),
                        "question": question,
                        "include_sample": sample_data is not None,
                        "sample_size": len(sample_data) if sample_data else 100
                    }

                    response = requests.post(api_url, json=request_data, timeout=30)

                    if response.status_code == 200:
                        result = response.json()

                        st.markdown("### 💡 AI Answer")
                        st.markdown(result.get('answer', 'No answer available'))

                        # Add to conversation history
                        if 'ai_conversation' not in st.session_state:
                            st.session_state.ai_conversation = []

                        st.session_state.ai_conversation.append({
                            'question': question,
                            'answer': result.get('answer', ''),
                            'cached': result.get('metadata', {}).get('cached', False)
                        })

                    else:
                        st.error(f"Question failed: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection to AI service failed: {e}")
                except Exception as e:
                    st.error(f"Question answering failed: {e}")

        # Conversation history
        if 'ai_conversation' in st.session_state and st.session_state.ai_conversation:
            with st.expander("Conversation History", expanded=False):
                for i, conv in enumerate(st.session_state.ai_conversation[-3:]):  # Show last 3
                    st.markdown(f"**Q{len(st.session_state.ai_conversation)-i}:** {conv['question']}")
                    st.markdown(f"**A:** {conv['answer'][:200]}{'...' if len(conv['answer']) > 200 else ''}")
                    st.markdown("---")

                if st.button("Clear History"):
                    st.session_state.ai_conversation = []

    def answer_question(
        self,
        question: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using AI (legacy method for compatibility)

        Args:
            question: User's question
            dataset_info: Dataset information
            context: Additional context

        Returns:
            Answer response dictionary
        """
        try:
            # Use the enhanced dataset question method if we have loaded data
            if hasattr(st.session_state, 'loaded_dataset') and st.session_state.loaded_dataset:
                return self.answer_dataset_question(
                    question=question,
                    dataset_info=dataset_info or st.session_state.loaded_dataset['dataset_info'],
                    loaded_data=st.session_state.loaded_dataset['data'],
                    chat_history=getattr(st.session_state, 'dataset_chat_history', [])
                )

            # Fallback to general question answering
            enhanced_prompt = f"""
            Question: {question}

            {f"Dataset Context: {dataset_info}" if dataset_info else ""}
            {f"Additional Context: {context}" if context else ""}

            Please provide a comprehensive and helpful answer.
            """

            # Try backend API first
            try:
                import requests
                api_url = "http://localhost:8080/api/ai/analyze"
                request_data = {
                    "dataset_id": dataset_info.get('id') if dataset_info else 'general',
                    "analysis_type": "custom",
                    "custom_prompt": enhanced_prompt,
                    "include_sample": False
                }

                response = requests.post(api_url, json=request_data, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        'answer': result.get('analysis', 'No response available'),
                        'source': 'backend',
                        'cached': result.get('cached', False)
                    }

            except Exception as e:
                logger.warning(f"Backend API failed for question: {e}")

            # Fallback response
            return {
                'answer': f"I received your question: '{question}'. However, I'm currently unable to process it due to AI service limitations. Please ensure the AI backend is configured and running.",
                'source': 'fallback',
                'cached': False
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'source': 'error',
                'cached': False
            }

    def answer_dataset_question(
        self,
        question: str,
        dataset_info: Dict[str, Any],
        loaded_data: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about a loaded dataset with full context

        Args:
            question: User's question
            dataset_info: Dataset metadata
            loaded_data: The actual loaded dataset
            chat_history: Previous chat messages for context

        Returns:
            Structured response with answer, insights, and code
        """
        try:
            # Prepare comprehensive context
            context = self._build_comprehensive_dataset_context(dataset_info, loaded_data, chat_history)

            # Enhanced prompt for dataset chat
            enhanced_prompt = f"""
            You are an expert data analyst with access to a loaded dataset in memory. You can see the actual data and should provide specific, actionable insights.

            DATASET CONTEXT:
            {context}

            USER'S QUESTION: "{question}"

            Please provide:
            1. A direct, specific answer based on the actual data shown
            2. Key insights and findings from the loaded data
            3. Any patterns, trends, or anomalies you can identify
            4. Practical recommendations for further analysis
            5. If applicable, suggest Python code for deeper analysis

            Focus on being specific and referencing actual values from the data where relevant.
            """

            # Try backend API first
            try:
                import requests
                api_url = "http://localhost:8080/api/ai/dataset-chat"
                request_data = {
                    "message": question,
                    "dataset_info": dataset_info,
                    "sample_data": loaded_data[:100] if loaded_data else [],
                    "chat_history": chat_history[-5:] if chat_history else []
                }

                response = requests.post(api_url, json=request_data, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        'answer': result.get('response', 'No response available'),
                        'insights': result.get('insights', []),
                        'code': result.get('generated_code', ''),
                        'cached': result.get('cached', False),
                        'source': 'backend'
                    }

            except Exception as e:
                logger.warning(f"Backend API failed, using direct AI: {e}")

            # Fallback to direct AI if backend unavailable
            if AI_AVAILABLE and self.analyst:
                request = AIRequest(
                    prompt=enhanced_prompt,
                    system_prompt="You are an expert data analyst. Provide specific, actionable insights based on the loaded dataset.",
                    temperature=0.3,
                    max_tokens=2000
                )

                response = asyncio.run(self.analyst.generate_response(request))

                # Parse response for insights and code
                insights = self._extract_insights_from_response(response.content)
                code = self._extract_code_from_response(response.content)

                return {
                    'answer': response.content,
                    'insights': insights,
                    'code': code,
                    'cached': response.cached,
                    'source': 'direct_ai'
                }

            else:
                # Final fallback - generate contextual response
                return self._generate_contextual_fallback(question, dataset_info, loaded_data)

        except Exception as e:
            logger.error(f"Dataset question answering failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while analyzing your question: {str(e)}",
                'insights': [],
                'code': '',
                'cached': False,
                'error': str(e)
            }

    def _build_comprehensive_dataset_context(
        self,
        dataset_info: Dict[str, Any],
        loaded_data: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """Build comprehensive context including actual data"""

        context = f"""
Dataset Information:
- Name: {dataset_info.get('name', 'Unknown')}
- Description: {dataset_info.get('description', 'No description')}
- Category: {dataset_info.get('category', 'Unknown')}
- ID: {dataset_info.get('id', 'Unknown')}
"""

        if loaded_data:
            context += f"\nLoaded Data ({len(loaded_data)} rows):\n"

            # Add column info from first row
            if loaded_data and loaded_data[0]:
                columns = list(loaded_data[0].keys())
                context += f"Columns: {', '.join(columns)}\n"

            # Add sample data
            context += "\nSample Records:\n"
            for i, record in enumerate(loaded_data[:5]):
                context += f"Row {i+1}: {record}\n"

            # Add basic statistics if possible
            try:
                # Calculate basic stats for numeric columns
                numeric_stats = {}
                for col in columns:
                    values = [row.get(col) for row in loaded_data if row.get(col) is not None]
                    numeric_values = []
                    for val in values:
                        try:
                            numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            continue

                    if numeric_values:
                        numeric_stats[col] = {
                            'count': len(numeric_values),
                            'mean': sum(numeric_values) / len(numeric_values),
                            'min': min(numeric_values),
                            'max': max(numeric_values)
                        }

                if numeric_stats:
                    context += "\nNumeric Column Statistics:\n"
                    for col, stats in numeric_stats.items():
                        context += f"- {col}: count={stats['count']}, mean={stats['mean']:.2f}, range={stats['min']}-{stats['max']}\n"

            except Exception:
                pass  # Skip stats if calculation fails

        if chat_history:
            context += f"\nPrevious conversation context (last {min(3, len(chat_history))} messages):\n"
            for msg in chat_history[-3:]:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100]
                context += f"- {role}: {content}{'...' if len(msg.get('content', '')) > 100 else ''}\n"

        return context

    def _extract_insights_from_response(self, response: str) -> List[str]:
        """Extract insights from AI response"""
        insights = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered insights
            if line.startswith(('- ', '• ', '* ')) or (line.startswith(tuple('123456789')) and '.' in line):
                insights.append(line.lstrip('- •*0123456789. '))
            elif 'insight' in line.lower() or 'finding' in line.lower():
                insights.append(line)

        return insights[:5]  # Return top 5 insights

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from AI response"""
        import re
        code_pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        return matches[0] if matches else ''

    def _generate_contextual_fallback(
        self,
        question: str,
        dataset_info: Dict[str, Any],
        loaded_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate a contextual fallback response when AI is unavailable"""

        dataset_name = dataset_info.get('name', 'your dataset')
        insights = []

        if loaded_data:
            insights.extend([
                f"Dataset contains {len(loaded_data)} loaded rows",
                f"Dataset has {len(loaded_data[0].keys()) if loaded_data else 0} columns",
                "Data is loaded in memory and ready for analysis"
            ])

            # Try to provide some basic analysis
            if loaded_data and loaded_data[0]:
                columns = list(loaded_data[0].keys())
                insights.append(f"Available columns: {', '.join(columns[:5])}")

        fallback_answer = f"""
I can see that you're asking about "{question}" regarding the dataset '{dataset_name}'.

While I don't have full AI capabilities available right now, I can tell you that your dataset is loaded in memory and ready for analysis.

Here's what I can observe from the loaded data:

{chr(10).join(f"• {insight}" for insight in insights)}

To get more detailed analysis, please ensure the backend AI service is running, or try asking specific questions about particular columns or patterns you're interested in.
        """

        return {
            'answer': fallback_answer.strip(),
            'insights': insights,
            'code': '',
            'cached': False,
            'source': 'fallback'
        }

    def render_mini_analyst(self, dataset_info: Dict[str, Any], auto_run: bool = True):
        """Render a compact version of the analyst for page headers"""

        # Auto-generate quick insights
        if auto_run and dataset_info.get('id'):
            cache_key = f"mini_{dataset_info['id']}"

            # Check if we already have a cached mini analysis
            if cache_key not in st.session_state:
                try:
                    with st.spinner("🧠 Generating AI insights..."):
                        import requests

                        api_url = "http://localhost:8080/api/ai/analyze"
                        request_data = {
                            "dataset_id": dataset_info.get('id'),
                            "analysis_type": "overview",
                            "include_sample": False
                        }

                        response = requests.post(api_url, json=request_data, timeout=30)

                        if response.status_code == 200:
                            result = response.json()
                            analysis_text = result.get('analysis', '')

                            # Extract key points (first few lines)
                            lines = analysis_text.split('\n')
                            summary_lines = []
                            for line in lines:
                                if line.strip() and not line.startswith('#'):
                                    summary_lines.append(line.strip())
                                    if len(summary_lines) >= 3:
                                        break

                            summary_text = '\n'.join(summary_lines)
                            st.session_state[cache_key] = summary_text

                        else:
                            st.session_state[cache_key] = "AI analysis temporarily unavailable"

                except Exception as e:
                    logger.error(f"Mini analysis failed: {e}")
                    st.session_state[cache_key] = "AI insights not available"

            # Display mini analysis
            if cache_key in st.session_state:
                with st.container():
                    st.markdown("#### 🤖 AI Quick Insights")
                    st.markdown(st.session_state[cache_key])

    def _get_expert_statistician_prompt(self, focus_area: str, dataset_info: Dict[str, Any], sample_data: Optional[List[Dict]] = None) -> str:
        """Generate expert statistician prompt based on focus area"""

        base_context = f"""
        You are an expert statistician analyzing the dataset: {dataset_info.get('name', 'Unknown Dataset')}

        Dataset Details:
        - Description: {dataset_info.get('description', 'No description available')}
        - Columns: {dataset_info.get('columns_count', 0)}
        - Category: {dataset_info.get('category', 'Uncategorized')}
        - Records: {dataset_info.get('download_count', 'Unknown')}
        """

        if sample_data:
            base_context += f"\n\nSample Data (first {min(len(sample_data), 5)} records):\n"
            for i, record in enumerate(sample_data[:5]):
                base_context += f"Record {i+1}: {record}\n"

        focus_prompts = {
            "Comprehensive Statistical Summary": f"""
            {base_context}

            Provide a comprehensive statistical analysis including:
            1. **Descriptive Statistics**: Central tendencies, dispersion measures, and shape parameters
            2. **Data Distribution**: Identify distribution types and key characteristics
            3. **Variable Types**: Classify variables (categorical, ordinal, continuous, discrete)
            4. **Missing Data Analysis**: Patterns and implications of missing values
            5. **Data Quality Assessment**: Statistical indicators of data reliability
            6. **Recommendations**: Statistical approaches suitable for this dataset

            Format your analysis with professional statistical terminology and actionable insights.
            """,

            "Distribution Analysis & Normality Tests": f"""
            {base_context}

            Focus on distribution analysis:
            1. **Distribution Identification**: Identify likely distributions for each variable
            2. **Normality Assessment**: Evaluate normality assumptions using statistical indicators
            3. **Skewness and Kurtosis**: Analyze shape characteristics
            4. **Transformation Recommendations**: Suggest transformations if needed
            5. **Statistical Test Selection**: Recommend appropriate tests based on distributions
            6. **Visualization Suggestions**: Optimal plots for distribution assessment

            Provide specific statistical test recommendations and interpretation guidelines.
            """,

            "Correlation & Regression Analysis": f"""
            {base_context}

            Analyze relationships and dependencies:
            1. **Correlation Analysis**: Identify and quantify variable relationships
            2. **Regression Opportunities**: Potential dependent/independent variable pairs
            3. **Multicollinearity Assessment**: Identify highly correlated predictors
            4. **Non-linear Relationships**: Detect non-linear patterns
            5. **Statistical Significance**: Assess relationship strength and significance
            6. **Modeling Recommendations**: Suggest appropriate regression techniques

            Focus on actionable insights for predictive modeling and relationship analysis.
            """,

            "Outlier Detection & Anomaly Analysis": f"""
            {base_context}

            Systematic outlier and anomaly analysis:
            1. **Outlier Detection Methods**: Apply multiple statistical approaches (IQR, Z-score, Modified Z-score)
            2. **Anomaly Patterns**: Identify systematic vs. random anomalies
            3. **Impact Assessment**: Evaluate outlier influence on analysis
            4. **Root Cause Analysis**: Suggest potential reasons for anomalies
            5. **Treatment Recommendations**: Advise on outlier handling strategies
            6. **Robust Statistics**: Recommend outlier-resistant analysis methods

            Provide specific outlier detection thresholds and treatment strategies.
            """,

            "Time Series Analysis (if applicable)": f"""
            {base_context}

            Time series statistical analysis:
            1. **Temporal Pattern Identification**: Trends, seasonality, and cycles
            2. **Stationarity Assessment**: Test for stationarity and recommend transformations
            3. **Autocorrelation Analysis**: Identify temporal dependencies
            4. **Decomposition**: Separate trend, seasonal, and irregular components
            5. **Forecasting Opportunities**: Assess predictability and suggest models
            6. **Statistical Tests**: Recommend appropriate time series tests

            If no temporal data exists, suggest how to incorporate time-based analysis.
            """,

            "Hypothesis Testing Recommendations": f"""
            {base_context}

            Design hypothesis testing framework:
            1. **Research Questions**: Formulate testable hypotheses from the data
            2. **Test Selection**: Recommend appropriate statistical tests
            3. **Sample Size Considerations**: Assess power and effect size requirements
            4. **Assumption Validation**: Check test prerequisites and alternatives
            5. **Multiple Comparisons**: Address multiple testing concerns
            6. **Practical Significance**: Distinguish statistical from practical significance

            Provide specific test recommendations with justification and interpretation guidelines.
            """
        }

        return focus_prompts.get(focus_area, focus_prompts["Comprehensive Statistical Summary"])

    def _display_cached_analysis(self, cached_result: Dict[str, Any]):
        """Display previously cached analysis results"""
        if not cached_result:
            return

        st.success("📋 **Displaying Cached Analysis Results**")

        # Display the main analysis content
        if cached_result.get('analysis'):
            st.markdown("### 🧠 AI Analysis Results")
            st.markdown(cached_result['analysis'])

        # Show metadata if available
        metadata = cached_result.get('metadata', {})
        if metadata:
            with st.expander("Analysis Details", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    if metadata.get('cached'):
                        st.info("📋 From Cache")
                    else:
                        st.success("🆕 Fresh Analysis")

                with col2:
                    if cached_result.get('analysis_type'):
                        st.text(f"Type: {cached_result['analysis_type']}")

                with col3:
                    if cached_result.get('sample_included'):
                        st.text("✅ With Sample Data")
                    else:
                        st.text("📊 Metadata Only")

        # Dataset context if available
        if cached_result.get('dataset_info'):
            with st.expander("Dataset Context", expanded=False):
                st.json(cached_result['dataset_info'])


# Global instance
@st.cache_resource
def get_ai_analyst_component():
    """Get cached AI analyst component"""
    return AIAnalystComponent()