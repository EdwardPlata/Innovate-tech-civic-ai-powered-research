"""
Codebase Agent Streamlit Component

Provides AI-powered codebase analysis, Q&A, and code suggestions
"""

import streamlit as st
import requests
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CodebaseAgentComponent:
    """Streamlit component for AI-powered codebase analysis"""

    def __init__(self):
        self.backend_url = "http://localhost:8080"

    def render_codebase_analyzer(self):
        """Render the main codebase analyzer interface"""
        st.header("🧠 AI Codebase Analyzer")
        st.caption("Analyze codebases with AI-powered insights and Q&A")

        # Check AI configuration
        if 'ai_config' not in st.session_state or not st.session_state.ai_config.get('api_key'):
            st.warning("⚠️ **AI Not Configured** - Please configure AI in the AI Setup page first.")
            if st.button("🚀 Go to AI Setup"):
                st.switch_page("pages/ai_setup.py")
            return

        # Codebase selection
        st.markdown("### 📁 Select Codebase")

        # Path input methods
        path_method = st.radio(
            "Choose how to specify the codebase path:",
            ["Current Directory", "Custom Path", "Browse"],
            horizontal=True
        )

        if path_method == "Current Directory":
            codebase_path = str(Path.cwd())
            st.info(f"📂 Using current directory: `{codebase_path}`")
        elif path_method == "Custom Path":
            codebase_path = st.text_input(
                "Enter codebase path:",
                value=str(Path.cwd()),
                help="Enter the full path to your codebase directory"
            )
        else:
            # Browse method - simplified file picker
            st.info("💡 Enter the path to your codebase directory below:")
            codebase_path = st.text_input(
                "Codebase path:",
                value=str(Path.cwd()),
                help="Full path to the directory containing your code"
            )

        # Validate path
        if not codebase_path or not Path(codebase_path).exists():
            if codebase_path:
                st.error(f"❌ Path does not exist: `{codebase_path}`")
            return

        if not Path(codebase_path).is_dir():
            st.error(f"❌ Path is not a directory: `{codebase_path}`")
            return

        st.success(f"✅ Valid codebase directory: `{codebase_path}`")

        # Analysis options
        st.markdown("### ⚙️ Analysis Options")

        col1, col2 = st.columns(2)

        with col1:
            analysis_focus = st.selectbox(
                "Analysis Focus:",
                ["overview", "architecture", "quality", "security"],
                format_func=lambda x: {
                    "overview": "🔍 Overview - General structure and patterns",
                    "architecture": "🏗️ Architecture - Design patterns and structure",
                    "quality": "⭐ Quality - Code quality and maintainability",
                    "security": "🔒 Security - Security analysis and vulnerabilities"
                }[x],
                help="Choose the focus of the AI analysis"
            )

        with col2:
            file_extensions = st.multiselect(
                "File Types to Analyze:",
                [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".cpp", ".c", ".h", ".rs", ".md"],
                default=[".py", ".js", ".ts", ".md"],
                help="Select which file types to include in analysis"
            )

        # Analysis buttons
        st.markdown("### 🚀 Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Analyze Codebase", type="primary"):
                self._run_codebase_analysis(codebase_path, analysis_focus, file_extensions)

        with col2:
            if st.button("📊 Get Statistics"):
                self._get_codebase_stats(codebase_path)

        with col3:
            if st.button("🧹 Clear Cache"):
                self._clear_analysis_cache()

        # Q&A Section
        st.markdown("### 💬 Ask Questions About Your Codebase")

        question = st.text_input(
            "Ask a question about your code:",
            placeholder="e.g., How does authentication work? Where is the main entry point?",
            help="Ask specific questions about your codebase structure, functions, or patterns"
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            context_files = st.text_input(
                "Focus on specific files (optional):",
                placeholder="e.g., auth.py, main.js, utils/",
                help="Comma-separated list of files or directories to focus the search on"
            )

        with col2:
            if st.button("🤔 Ask AI", disabled=not question):
                context_file_list = [f.strip() for f in context_files.split(",")] if context_files else None
                self._answer_codebase_question(codebase_path, question, context_file_list)

        # Recent analyses
        self._show_recent_analyses()

    def _run_codebase_analysis(self, codebase_path: str, analysis_focus: str, file_extensions: List[str]):
        """Run AI-powered codebase analysis"""

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔧 Preparing codebase analysis...")
            progress_bar.progress(0.1)

            # Prepare request
            request_data = {
                "codebase_path": codebase_path,
                "analysis_focus": analysis_focus,
                "file_extensions": file_extensions if file_extensions else None
            }

            status_text.text("📡 Sending analysis request...")
            progress_bar.progress(0.3)

            # Make API request
            response = requests.post(
                f"{self.backend_url}/api/codebase/analyze",
                json=request_data,
                timeout=300  # 5 minutes for large codebases
            )

            status_text.text("🤖 AI is analyzing your codebase...")
            progress_bar.progress(0.8)

            if response.status_code == 200:
                result = response.json()

                status_text.text("✅ Analysis complete!")
                progress_bar.progress(1.0)

                # Display results
                self._display_analysis_results(result)

                # Store in session for history
                if 'codebase_analyses' not in st.session_state:
                    st.session_state.codebase_analyses = []

                st.session_state.codebase_analyses.insert(0, {
                    'codebase_path': codebase_path,
                    'analysis_focus': analysis_focus,
                    'result': result,
                    'timestamp': result.get('timestamp')
                })

                # Keep only last 5 analyses
                st.session_state.codebase_analyses = st.session_state.codebase_analyses[:5]

            else:
                status_text.text("❌ Analysis failed")
                st.error(f"Analysis failed: {response.status_code} - {response.text}")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.empty()
            st.error("⏱️ **Analysis Timeout** - Large codebases may take several minutes to analyze.")

            with st.expander("🔧 Timeout Troubleshooting"):
                st.markdown("""
                **Try these solutions:**
                1. **Reduce scope** - Analyze specific file types only
                2. **Exclude large directories** - Skip build/, node_modules/, etc.
                3. **Use smaller codebase** - Start with a subset of your project
                4. **Check backend** - Ensure the AI backend is running properly
                """)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis failed: {str(e)}")
            logger.error(f"Codebase analysis error: {e}")

    def _answer_codebase_question(self, codebase_path: str, question: str, context_files: Optional[List[str]]):
        """Answer a question about the codebase"""

        with st.spinner("🤔 AI is analyzing your codebase to answer your question..."):
            try:
                request_data = {
                    "codebase_path": codebase_path,
                    "question": question,
                    "context_files": context_files
                }

                response = requests.post(
                    f"{self.backend_url}/api/codebase/question",
                    json=request_data,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    self._display_question_answer(result)

                    # Store in conversation history
                    if 'codebase_conversations' not in st.session_state:
                        st.session_state.codebase_conversations = []

                    st.session_state.codebase_conversations.insert(0, {
                        'question': question,
                        'answer': result,
                        'codebase_path': codebase_path
                    })

                    # Keep only last 10 conversations
                    st.session_state.codebase_conversations = st.session_state.codebase_conversations[:10]

                else:
                    st.error(f"Question failed: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Question answering failed: {str(e)}")
                logger.error(f"Codebase question error: {e}")

    def _get_codebase_stats(self, codebase_path: str):
        """Get and display codebase statistics"""

        with st.spinner("📊 Gathering codebase statistics..."):
            try:
                # URL encode the path
                import urllib.parse
                encoded_path = urllib.parse.quote(codebase_path, safe='')

                response = requests.get(
                    f"{self.backend_url}/api/codebase/stats/{encoded_path}",
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    self._display_codebase_stats(result)
                else:
                    st.error(f"Statistics failed: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Statistics gathering failed: {str(e)}")
                logger.error(f"Codebase stats error: {e}")

    def _display_analysis_results(self, result: Dict[str, Any]):
        """Display codebase analysis results"""

        st.markdown("### 🧠 AI Codebase Analysis Results")

        # Analysis metadata
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Files Analyzed", result.get('metadata', {}).get('files_analyzed', 0))

        with col2:
            st.metric("Code Chunks", result.get('chunk_count', 0))

        with col3:
            focus = result.get('analysis_focus', 'unknown').title()
            st.metric("Analysis Focus", focus)

        with col4:
            cached = "✅ Cached" if result.get('cached') else "🆕 Fresh"
            st.metric("Analysis Type", cached)

        # Main analysis content
        analysis_content = result.get('analysis', '')
        if analysis_content:
            st.markdown("#### 📝 Analysis Report")
            st.markdown(analysis_content)
        else:
            st.warning("No analysis content available")

        # Codebase statistics
        stats = result.get('codebase_stats', {})
        if stats and not stats.get('error'):
            with st.expander("📊 Detailed Statistics", expanded=False):
                self._display_stats_details(stats)

        # Technical details
        with st.expander("🔧 Technical Details", expanded=False):
            st.json({
                "provider_used": result.get('provider_used'),
                "timestamp": result.get('timestamp'),
                "codebase_path": result.get('codebase_path'),
                "metadata": result.get('metadata', {})
            })

    def _display_question_answer(self, result: Dict[str, Any]):
        """Display Q&A results"""

        st.markdown("### 💡 AI Answer")

        # Question context
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Question:** {result.get('question', '')}")

        with col2:
            chunks_analyzed = result.get('chunks_analyzed', 0)
            st.metric("Chunks Analyzed", chunks_analyzed)

        # Answer content
        answer = result.get('answer', '')
        if answer:
            st.markdown("#### 📝 Answer")
            st.markdown(answer)
        else:
            st.warning("No answer available")

        # Relevant files
        relevant_files = result.get('relevant_files', [])
        if relevant_files:
            with st.expander(f"📁 Relevant Files ({len(relevant_files)})", expanded=False):
                for file_path in relevant_files[:10]:  # Show top 10
                    st.code(file_path)

        # Technical details
        with st.expander("🔧 Answer Details", expanded=False):
            st.json({
                "search_scope": result.get('search_scope'),
                "provider_used": result.get('provider_used'),
                "cached": result.get('cached'),
                "timestamp": result.get('timestamp')
            })

    def _display_codebase_stats(self, result: Dict[str, Any]):
        """Display codebase statistics"""

        st.markdown("### 📊 Codebase Statistics")

        stats = result.get('statistics', {})
        if stats.get('error'):
            st.error(f"Statistics error: {stats['error']}")
            return

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Files", stats.get('total_files', 0))

        with col2:
            st.metric("Code Chunks", stats.get('total_chunks', 0))

        with col3:
            st.metric("Functions", stats.get('chunk_types', {}).get('function', 0))

        with col4:
            st.metric("Classes", stats.get('chunk_types', {}).get('class', 0))

        self._display_stats_details(stats)

    def _display_stats_details(self, stats: Dict[str, Any]):
        """Display detailed statistics"""

        col1, col2 = st.columns(2)

        # File types distribution
        with col1:
            st.markdown("#### 📄 File Types")
            file_extensions = stats.get('file_extensions', {})
            if file_extensions:
                for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{ext or 'No extension'}**: {count} files")
            else:
                st.write("No file type data available")

        # Chunk types distribution
        with col2:
            st.markdown("#### 🧩 Code Components")
            chunk_types = stats.get('chunk_types', {})
            if chunk_types:
                for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                    icon = {"function": "🔧", "class": "📦", "module": "📄", "config": "⚙️"}.get(chunk_type, "📝")
                    st.write(f"**{icon} {chunk_type.title()}**: {count}")
            else:
                st.write("No component data available")

        # Largest files
        largest_files = stats.get('largest_files', [])
        if largest_files:
            st.markdown("#### 📈 Largest Files")
            for file_path, size in largest_files[:5]:
                # Get relative path for display
                display_path = Path(file_path).name if len(file_path) > 50 else file_path
                st.write(f"**{display_path}**: {size:,} characters")

    def _show_recent_analyses(self):
        """Show recent codebase analyses"""

        # Recent analyses
        if 'codebase_analyses' in st.session_state and st.session_state.codebase_analyses:
            with st.expander(f"📚 Recent Analyses ({len(st.session_state.codebase_analyses)})", expanded=False):
                for i, analysis in enumerate(st.session_state.codebase_analyses):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        path = analysis['codebase_path']
                        display_path = Path(path).name if len(path) > 30 else path
                        st.write(f"**{display_path}**")

                    with col2:
                        st.write(analysis['analysis_focus'].title())

                    with col3:
                        if st.button("View", key=f"view_analysis_{i}"):
                            self._display_analysis_results(analysis['result'])

        # Conversation history
        if 'codebase_conversations' in st.session_state and st.session_state.codebase_conversations:
            with st.expander(f"💬 Recent Questions ({len(st.session_state.codebase_conversations)})", expanded=False):
                for i, conv in enumerate(st.session_state.codebase_conversations):
                    question = conv['question']
                    display_question = question[:60] + "..." if len(question) > 60 else question

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Q**: {display_question}")

                    with col2:
                        if st.button("View Answer", key=f"view_conv_{i}"):
                            self._display_question_answer(conv['answer'])

    def _clear_analysis_cache(self):
        """Clear analysis cache"""
        try:
            response = requests.delete(f"{self.backend_url}/api/cache/clear?cache_type=codebase_analysis")

            if response.status_code == 200:
                st.success("✅ Analysis cache cleared successfully")

                # Also clear session state
                if 'codebase_analyses' in st.session_state:
                    del st.session_state.codebase_analyses
                if 'codebase_conversations' in st.session_state:
                    del st.session_state.codebase_conversations

            else:
                st.error(f"Cache clear failed: {response.status_code}")

        except Exception as e:
            st.error(f"Cache clear failed: {str(e)}")


# Global instance
@st.cache_resource
def get_codebase_agent_component():
    """Get cached codebase agent component"""
    return CodebaseAgentComponent()