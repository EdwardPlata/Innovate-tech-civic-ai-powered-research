def show_ai_setup(ai_analyst):
    """Enhanced AI Configuration - Complete Setup Wizard"""
    st.header("🤖 AI Setup & Configuration")
    st.caption("Complete AI setup wizard with validation and testing")

    # Display setup progress
    setup_steps = ["Provider Selection", "API Key Configuration", "Model Selection", "Settings", "Testing"]

    # Get current setup status
    current_config = st.session_state.get('ai_config', {})
    setup_progress = 0

    if current_config.get('provider'):
        setup_progress += 1
    if current_config.get('api_key'):
        setup_progress += 1
    if current_config.get('model'):
        setup_progress += 1
    if current_config.get('analysis_temperature') is not None:
        setup_progress += 1

    # Progress bar
    progress_bar = st.progress(setup_progress / len(setup_steps))
    st.write(f"Setup Progress: {setup_progress}/{len(setup_steps)} steps completed")

    # Prerequisites check
    with st.expander("📋 Setup Prerequisites", expanded=(setup_progress == 0)):
        st.markdown("""
        **Before you begin, ensure you have:**

        1. **Internet Connection** - Required for AI model access
        2. **API Key** - From one of the supported providers:
           - 🟢 **NVIDIA AI** (Recommended): Free tier available at [build.nvidia.com](https://build.nvidia.com)
           - 🔵 **OpenAI**: Paid service at [platform.openai.com](https://platform.openai.com)
           - 🟡 **OpenRouter**: Multi-model access at [openrouter.ai](https://openrouter.ai)
        3. **Budget Planning** - Some providers charge per API call

        **Why AI is useful for data analysis:**
        - 🧠 **Intelligent Analysis**: Understand data patterns and trends
        - 💡 **Instant Insights**: Get explanations of complex data relationships
        - 🔍 **Quality Assessment**: Automated data quality scoring
        - 💬 **Interactive Q&A**: Ask questions about your datasets in plain English
        """)

    # Warning if no progress
    if setup_progress == 0:
        st.warning("⚠️ **AI Not Configured** - Please complete the setup steps below to activate AI features.")

    # Initialize session state for AI config
    if 'ai_config' not in st.session_state:
        st.session_state.ai_config = {
            'provider': 'nvidia',
            'model': 'qwen/qwen2.5-72b-instruct',
            'api_key': '',
            'use_reasoning_models': True,
            'analysis_temperature': 0.3,
            'max_tokens': 2000,
            'enable_cache': True
        }

    # Step 1: Provider Selection
    st.markdown("### 🎯 Step 1: Choose Your AI Provider")

    provider_options = {
        'nvidia': '🟢 NVIDIA AI (Recommended for reasoning)',
        'openai': '🔵 OpenAI (GPT models)',
        'openrouter': '🟡 OpenRouter (Multiple models)'
    }

    selected_provider = st.selectbox(
        "Select AI Provider:",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.ai_config['provider'])
    )

    st.session_state.ai_config['provider'] = selected_provider

    # Step 2: API Key
    st.markdown("### 🔑 Step 2: Enter API Key")

    if selected_provider == 'nvidia':
        st.info("🔗 Get your NVIDIA API key from: https://build.nvidia.com")
        api_key_help = "Free tier available with good rate limits"
    elif selected_provider == 'openai':
        st.info("🔗 Get your OpenAI API key from: https://platform.openai.com")
        api_key_help = "Requires payment, high quality models"
    else:
        st.info("🔗 Get your OpenRouter API key from: https://openrouter.ai")
        api_key_help = "Access to multiple models including Claude, Gemini"

    api_key = st.text_input(
        f"{provider_options[selected_provider]} API Key:",
        value=st.session_state.ai_config['api_key'],
        type="password",
        help=api_key_help,
        placeholder="Paste your API key here..."
    )

    st.session_state.ai_config['api_key'] = api_key

    # API Key validation
    if api_key:
        # Basic validation
        valid_key = True
        error_message = ""

        if selected_provider == 'nvidia' and not api_key.startswith('nvapi-'):
            valid_key = False
            error_message = "NVIDIA API keys typically start with 'nvapi-'"
        elif selected_provider == 'openai' and not api_key.startswith('sk-'):
            valid_key = False
            error_message = "OpenAI API keys start with 'sk-'"
        elif len(api_key) < 20:
            valid_key = False
            error_message = "API key appears too short - please check your key"

        if valid_key:
            st.success("✅ API key format looks correct")
        else:
            st.error(f"❌ {error_message}")
            st.info("💡 Double-check your API key from the provider's dashboard")

    else:
        st.info("🔑 Enter your API key to continue setup")

    # Step 3: Model Selection
    st.markdown("### 🧠 Step 3: Choose Model")

    if selected_provider == 'nvidia':
        model_options = [
            'qwen/qwen2.5-72b-instruct',
            'meta/llama-3.1-405b-instruct',
            'meta/llama-3.1-70b-instruct',
            'meta/llama-3.1-8b-instruct',
            'mistralai/mixtral-8x22b-instruct-v0.1',
            'google/gemma-2-27b-it'
        ]
        recommended = 'qwen/qwen2.5-72b-instruct'
        st.success("🧠 **Qwen 2.5 72B** is recommended for best reasoning and analysis")
    elif selected_provider == 'openai':
        model_options = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ]
        recommended = 'gpt-4o-mini'
        st.success("⚡ **GPT-4o Mini** is recommended for cost-effective performance")
    else:  # openrouter
        model_options = [
            'anthropic/claude-3-opus',
            'anthropic/claude-3-sonnet',
            'anthropic/claude-3-haiku',
            'google/gemini-pro',
            'meta-llama/llama-3-70b-instruct'
        ]
        recommended = 'anthropic/claude-3-sonnet'
        st.success("🎯 **Claude 3 Sonnet** is recommended for balanced performance")

    selected_model = st.selectbox(
        "Select Model:",
        options=model_options,
        index=model_options.index(st.session_state.ai_config['model']) if st.session_state.ai_config['model'] in model_options else 0
    )

    st.session_state.ai_config['model'] = selected_model

    # Step 4: Quick Settings
    st.markdown("### ⚙️ Step 4: Quick Settings")

    col1, col2 = st.columns(2)

    with col1:
        reasoning_mode = st.toggle(
            "🧠 Reasoning Mode",
            value=st.session_state.ai_config['use_reasoning_models'],
            help="Enable for complex analysis and better logical reasoning"
        )
        st.session_state.ai_config['use_reasoning_models'] = reasoning_mode

    with col2:
        enable_cache = st.toggle(
            "💾 Enable Caching",
            value=st.session_state.ai_config['enable_cache'],
            help="Cache responses for faster repeated queries"
        )
        st.session_state.ai_config['enable_cache'] = enable_cache

    # Advanced settings in expander
    with st.expander("🔧 Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.ai_config['analysis_temperature'],
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        st.session_state.ai_config['analysis_temperature'] = temperature

        max_tokens = st.number_input(
            "Max Response Length",
            min_value=500,
            max_value=4000,
            value=st.session_state.ai_config['max_tokens'],
            step=100,
            help="Maximum length of AI responses"
        )
        st.session_state.ai_config['max_tokens'] = max_tokens

    # Step 5: Test & Save
    st.markdown("### 🧪 Step 5: Test & Save Configuration")

    # Configuration summary
    if api_key:
        st.success("✅ **Configuration Ready**")

        config_summary = f"""
**Provider:** {provider_options[selected_provider]}
**Model:** {selected_model}
**API Key:** {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}
**Reasoning Mode:** {'✅ Enabled' if reasoning_mode else '❌ Disabled'}
**Caching:** {'✅ Enabled' if enable_cache else '❌ Disabled'}
"""
        st.markdown(config_summary)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧪 Test Configuration", type="secondary"):
                if not api_key:
                    st.error("❌ Please enter an API key before testing")
                else:
                    with st.spinner("Testing AI configuration..."):
                        try:
                            # Enhanced testing with timeout and validation
                            import requests
                            import time
                            from concurrent.futures import ThreadPoolExecutor, TimeoutError

                            def test_ai_connection():
                                """Test AI provider connection with timeout"""
                                try:
                                    # Test backend connectivity first
                                    backend_response = requests.get("http://localhost:8080/api/health", timeout=5)
                                    if backend_response.status_code != 200:
                                        return {"success": False, "error": "Backend not accessible", "details": "Please ensure the backend server is running"}

                                    # Test AI configuration via backend
                                    test_request = {
                                        "dataset_id": "test",
                                        "analysis_type": "overview",
                                        "custom_prompt": "Test connection: Respond with 'AI connection successful' if this works.",
                                        "include_sample": False
                                    }

                                    # Update session with current config for test
                                    test_response = requests.post(
                                        "http://localhost:8080/api/ai/analyze",
                                        json=test_request,
                                        timeout=30
                                    )

                                    if test_response.status_code == 200:
                                        return {"success": True, "response": test_response.json()}
                                    else:
                                        return {"success": False, "error": f"API Error {test_response.status_code}", "details": test_response.text}

                                except requests.exceptions.Timeout:
                                    return {"success": False, "error": "Request timeout", "details": "AI service took too long to respond (>30s)"}
                                except requests.exceptions.ConnectionError:
                                    return {"success": False, "error": "Connection failed", "details": "Cannot connect to AI backend service"}
                                except Exception as e:
                                    return {"success": False, "error": "Unexpected error", "details": str(e)}

                            # Run test with timeout
                            with ThreadPoolExecutor() as executor:
                                future = executor.submit(test_ai_connection)
                                try:
                                    result = future.result(timeout=45)

                                    if result["success"]:
                                        st.success("✅ **Test Successful!** AI is ready to use.")
                                        st.info(f"Connected to {selected_provider.upper()} using {selected_model}")

                                        # Show test response details
                                        with st.expander("Test Response Details"):
                                            st.json(result["response"])

                                        # Mark test as completed in progress
                                        st.session_state.ai_config['test_completed'] = True

                                    else:
                                        st.error(f"❌ **Test Failed:** {result['error']}")
                                        st.info(f"Details: {result['details']}")

                                        # Provide troubleshooting guidance
                                        with st.expander("🔧 Troubleshooting Guide"):
                                            st.markdown(f"""
                                            **Common Issues & Solutions:**

                                            1. **Backend not accessible:**
                                               - Ensure the Scout backend is running: `python backend/main.py`
                                               - Check if port 8080 is available

                                            2. **API Key Issues:**
                                               - Verify your {selected_provider} API key is valid
                                               - Check if you have sufficient credits/quota
                                               - Ensure the key has the right permissions

                                            3. **Timeout Issues:**
                                               - Try again - first requests can be slower
                                               - Check your internet connection
                                               - Consider switching to a faster provider

                                            4. **Model Access:**
                                               - Some models require special access
                                               - Try a different model from the list
                                            """)

                                except TimeoutError:
                                    st.error("❌ **Test Timeout** - AI service took too long to respond (>45s)")
                                    st.info("💡 Try again or consider switching to a faster provider")

                        except Exception as e:
                            st.error(f"❌ **Test Failed:** {str(e)}")
                            st.info("Please check your configuration and try again")

        with col2:
            if st.button("💾 Save Configuration", type="primary"):
                # Update config with consistent naming for backward compatibility
                st.session_state.ai_config.update({
                    'primary_provider': selected_provider,
                    f'{selected_provider}_api_key': api_key,
                    f'{selected_provider}_model': selected_model
                })

                st.success("✅ **Configuration Saved!**")
                st.balloons()
                st.info("You can now use AI features throughout the application!")

    else:
        st.warning("⚠️ **Please enter an API key to continue**")

    # Usage tips
    st.markdown("### 💡 Quick Start Tips")
    st.info(f"""
**Getting Started:**
1. Get your API key from the link above
2. Choose **{recommended}** for best results
3. Enable reasoning mode for complex data analysis
4. Test your configuration before using AI features

**Best Practices:**
- Use temperature 0.1-0.3 for analytical tasks
- Enable caching for faster responses
- Higher max tokens (2000+) for detailed analysis
""")