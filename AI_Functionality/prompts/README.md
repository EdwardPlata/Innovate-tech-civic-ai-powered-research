# 🧠 AI Prompts - Professional Statistician System

## Overview

The AI Prompts system provides sophisticated prompt templates that transform NVIDIA AI models into domain-specific experts. The flagship implementation is the **NVIDIA Professional Statistician** - an AI persona with PhD-level expertise in statistics, pandas, and plotly.

## 🎯 NVIDIA Statistician Features

### Professional Persona
- **Dr. Elena Rodriguez**: Senior Data Statistician with 15+ years experience
- **Stanford PhD in Statistics**
- **Former Principal Data Scientist at Google and Netflix**
- **40+ published research papers**

### Core Capabilities
- **Natural Language Data Exploration**: Ask questions about datasets in plain English
- **Professional Statistical Analysis**: Hypothesis testing, correlation analysis, regression
- **Advanced Visualizations**: Interactive plotly charts with statistical annotations
- **Dataset Joining Intelligence**: AI-guided data integration with quality assessment
- **Comprehensive EDA**: Automated exploratory data analysis
- **Statistical Testing**: Hypothesis testing with proper interpretation

### Generated Code Quality
- **Production-ready pandas code** for data manipulation
- **Professional plotly visualizations** with statistical rigor
- **Comprehensive error handling** and data validation
- **Statistical best practices** and proper methodology
- **Clear documentation** and explanations

## 📊 Analysis Types Available

### 1. Natural Language Exploration
```python
# Example usage
question = "What are the key factors that influence customer churn? Show me correlations and statistical tests."

# Generates:
# - Comprehensive statistical analysis
# - Correlation matrices with significance tests
# - Professional visualizations
# - Actionable business insights
```

### 2. Professional Visualizations
```python
# Example request
chart_request = "Create an interactive dashboard showing customer segments with statistical comparisons"

# Generates:
# - Interactive plotly dashboards
# - Statistical annotations
# - Professional styling
# - Export-ready formats
```

### 3. Intelligent Dataset Joining
```python
# Example objective
join_objective = "Combine customer demographics with purchase history for churn analysis"

# Generates:
# - Join strategy assessment
# - Data quality validation
# - Referential integrity checks
# - Complete pandas implementation
```

### 4. Statistical Hypothesis Testing
```python
# Example hypothesis
hypothesis = "Customer satisfaction scores are significantly different between product categories"

# Generates:
# - Proper hypothesis formulation (H₀, H₁)
# - Appropriate test selection
# - Power analysis and effect sizes
# - Professional interpretation
```

### 5. Comprehensive EDA
```python
# Automated professional EDA
focus_areas = ["distributions", "correlations", "outliers", "time_series"]

# Generates:
# - Multi-phase statistical analysis
# - Professional-quality visualizations
# - Business insights and recommendations
# - Executive summary for stakeholders
```

## 🏗️ Architecture

### Prompt System Components
```
AI_Functionality/prompts/
├── __init__.py                    # Package initialization
├── nvidia_statistician.py         # NVIDIA statistician prompt system
└── README.md                      # This documentation
```

### Integration Points
- **DataExplorer**: Main interface for AI-powered analysis
- **Backend API**: RESTful endpoints for web integration
- **Streamlit Frontend**: User-friendly interface
- **Multi-provider Support**: NVIDIA preferred, OpenAI/OpenRouter fallback

## 🚀 Usage Examples

### Basic Dataset Exploration
```python
from AI_Functionality.core.data_explorer import DataExplorer
from AI_Functionality import DataAnalyst

# Initialize with NVIDIA preference
analyst = DataAnalyst(nvidia_api_key="nvapi-your-key")
explorer = DataExplorer(analyst, prefer_nvidia=True)

# Explore dataset with natural language
result = await explorer.explore_dataset(
    dataframe=df,
    dataset_name="Customer Analytics",
    user_question="What drives customer lifetime value? Show me statistical relationships and create visualizations."
)

# Get professional analysis with pandas/plotly code
print(result['sections']['executive_summary'])
print(result['sections']['python_code'])
```

### Advanced Statistical Testing
```python
# Perform professional hypothesis testing
result = await explorer.perform_statistical_test(
    dataframe=df,
    dataset_name="A/B Test Results",
    hypothesis="Treatment group has significantly higher conversion rates",
    test_type="two_sample_ttest"
)

# Get complete statistical analysis
print(result['sections']['methodology'])  # Statistical approach
print(result['sections']['python_code'])  # Implementation code
print(result['sections']['insights'])     # Professional interpretation
```

### Intelligent Dataset Joining
```python
# Join datasets with AI guidance
result = await explorer.join_datasets(
    primary_dataframe=customers_df,
    datasets_to_join={"transactions": transactions_df, "products": products_df},
    join_objective="Combine customer, transaction, and product data for churn analysis",
    analysis_goal="Build a predictive model for customer churn"
)

# Get join strategy and implementation
print(result['sections']['methodology'])   # Join strategy
print(result['sections']['python_code'])   # Pandas join code
```

## 🎨 Visualization Capabilities

### Professional Chart Types
- **Statistical Plots**: Box plots with significance tests, violin plots with distributions
- **Correlation Analysis**: Heatmaps with statistical significance annotations
- **Time Series**: Trend analysis with confidence intervals and seasonality detection
- **Distribution Analysis**: Histograms with distribution fitting and normality tests
- **Comparative Analysis**: Multi-group comparisons with statistical tests
- **Interactive Dashboards**: Multi-panel dashboards with cross-filtering

### Visualization Features
- **Statistical Annotations**: Confidence intervals, p-values, effect sizes
- **Professional Styling**: Publication-ready appearance
- **Interactive Elements**: Hover details, zoom, selection, filtering
- **Export Options**: HTML, PNG, PDF, SVG formats
- **Accessibility**: Color-blind friendly palettes, proper contrast

## 🧪 Statistical Rigor

### Professional Standards
- **Assumption Checking**: Validates statistical test assumptions
- **Effect Size Reporting**: Provides practical significance assessment
- **Confidence Intervals**: Includes uncertainty quantification
- **Multiple Testing Correction**: Handles family-wise error rates
- **Power Analysis**: Assesses statistical power and sample size adequacy

### Methodology Documentation
- **Clear Explanations**: Explains statistical concepts for business users
- **Limitation Discussion**: Acknowledges analysis limitations and caveats
- **Recommendation Format**: Provides actionable business recommendations
- **Follow-up Suggestions**: Suggests additional analyses to consider

## 🔧 Configuration

### NVIDIA API Setup
```python
# Recommended configuration for statistician
analyst = DataAnalyst(
    nvidia_api_key="nvapi-your-key",
    nvidia_model="qwen/qwen2.5-72b-instruct",  # Excellent for reasoning
    primary_provider="nvidia",
    fallback_providers=["openai", "openrouter"]
)
```

### Advanced Options
```python
# Custom statistician configuration
explorer = DataExplorer(
    ai_analyst=analyst,
    prefer_nvidia=True,                 # Use NVIDIA's reasoning capabilities
    fallback_providers=["openai"]       # High-quality fallback
)
```

## 📈 Performance Benefits

### NVIDIA Advantages
- **Free Tier Available**: No cost for development and testing
- **Advanced Reasoning**: Excellent for complex statistical analysis
- **Mathematical Capabilities**: Strong performance on statistical computations
- **Code Generation**: High-quality pandas and plotly code

### Quality Assurance
- **Consistent Output**: Professional-grade analysis every time
- **Validated Methods**: Uses established statistical procedures
- **Error Handling**: Robust error handling and validation
- **Documentation**: Comprehensive code documentation

## 🚀 Integration Ready

### Backend API Endpoints
- `POST /api/data-explorer/explore` - Natural language exploration
- `POST /api/data-explorer/visualize` - Professional visualizations
- `POST /api/data-explorer/join` - Intelligent dataset joining
- `POST /api/data-explorer/statistical-test` - Hypothesis testing
- `POST /api/data-explorer/comprehensive-eda` - Full EDA analysis

### Frontend Components
- **Streamlit Interface**: User-friendly data exploration
- **File Upload**: CSV data loading and management
- **Interactive Results**: Execute and modify generated code
- **Analysis History**: Track and revisit previous analyses

---

The NVIDIA Professional Statistician system transforms AI into a PhD-level data expert, providing production-quality analysis with the rigor expected in professional statistical consulting.

**Ready for immediate use with any tabular dataset! 🎉**