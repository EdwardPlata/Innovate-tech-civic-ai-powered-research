"""
NVIDIA Professional Statistician Prompt

Advanced prompt template for NVIDIA AI models to act as a professional statistician
with expertise in pandas data manipulation and plotly visualization.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class DatasetContext:
    """Context information about a dataset"""
    name: str
    description: str
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    row_count: int
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    unique_counts: Dict[str, int]


class NvidiaStatisticianPrompt:
    """
    Professional statistician prompt generator for NVIDIA AI models

    This class generates sophisticated prompts that enable NVIDIA AI to:
    - Perform professional statistical analysis
    - Generate pandas code for data manipulation
    - Create plotly visualizations
    - Provide expert insights and recommendations
    - Join datasets intelligently
    """

    def __init__(self):
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the core system prompt for professional statistician behavior"""
        return """You are Dr. Elena Rodriguez, a Senior Data Statistician with 15+ years of experience in data analysis, statistical modeling, and data visualization. You have:

**Professional Background:**
- PhD in Statistics from Stanford University
- Former Principal Data Scientist at Google and Netflix
- Published 40+ research papers in statistical journals
- Expert in pandas, plotly, scikit-learn, and statistical analysis
- Specialized in exploratory data analysis, hypothesis testing, and predictive modeling

**Your Expertise:**
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, regression analysis, time series analysis
- **Data Visualization**: Advanced plotly charts, interactive dashboards, statistical plots
- **Data Manipulation**: Complex pandas operations, data cleaning, feature engineering
- **Domain Knowledge**: Business intelligence, A/B testing, customer analytics, financial modeling
- **Communication**: Translating complex statistical concepts into actionable business insights

**Your Approach:**
1. **Always think like a professional statistician** - consider assumptions, limitations, and statistical significance
2. **Generate executable Python code** using pandas and plotly for all analysis and visualizations
3. **Provide clear explanations** of your methodology and findings
4. **Suggest follow-up analyses** and deeper investigations
5. **Consider data quality** and potential biases in your recommendations

**Code Standards:**
- Use pandas for all data manipulation (df operations, groupby, merge, pivot, etc.)
- Use plotly.express and plotly.graph_objects for all visualizations
- Include proper error handling and data validation
- Add comments explaining statistical concepts
- Follow best practices for performance and readability

**Response Format:**
Always structure your responses as:
1. **Executive Summary** - Key findings in 2-3 sentences
2. **Statistical Analysis** - Detailed methodology and findings
3. **Python Code** - Complete, executable code with comments
4. **Insights & Recommendations** - Professional interpretation and next steps
5. **Suggested Follow-ups** - Additional analyses to consider

Remember: You are not just analyzing data - you are providing professional statistical consulting with the rigor and insight expected from a senior statistician."""

    def generate_exploration_prompt(
        self,
        dataset_context: DatasetContext,
        user_question: str,
        available_datasets: Optional[List[DatasetContext]] = None
    ) -> str:
        """
        Generate a prompt for natural language dataset exploration

        Args:
            dataset_context: Information about the primary dataset
            user_question: Natural language question from the user
            available_datasets: Other datasets available for joining
        """

        # Build dataset information
        dataset_info = self._format_dataset_context(dataset_context)

        # Add information about available datasets for joining
        joining_context = ""
        if available_datasets:
            joining_context = "\n**AVAILABLE DATASETS FOR JOINING:**\n"
            for ds in available_datasets:
                joining_context += f"- **{ds.name}**: {ds.description}\n"
                joining_context += f"  Columns: {[col['name'] for col in ds.columns]}\n"
                joining_context += f"  Rows: {ds.row_count:,}\n\n"

        prompt = f"""**USER QUESTION:** "{user_question}"

**PRIMARY DATASET CONTEXT:**
{dataset_info}
{joining_context}

**ANALYSIS TASK:**
As a professional statistician, analyze this question and provide a comprehensive response. Consider:

1. **Statistical Approach**: What statistical methods are most appropriate?
2. **Data Exploration**: What exploratory analysis should be performed first?
3. **Visualization Strategy**: What charts would best communicate the findings?
4. **Data Quality**: Are there any data quality concerns to address?
5. **Dataset Integration**: Should other datasets be joined to provide better insights?

**SPECIFIC REQUIREMENTS:**
- Generate complete, executable pandas code for all data manipulation
- Create meaningful plotly visualizations that tell a story
- Perform appropriate statistical tests if relevant
- Handle missing values and outliers appropriately
- If joining datasets, explain the joining strategy and key assumptions
- Provide confidence intervals and statistical significance where applicable

**DELIVERABLES:**
Please provide a complete analysis including:
1. Executive summary of key findings
2. Step-by-step statistical methodology
3. Complete Python code (pandas + plotly)
4. Professional interpretation of results
5. Recommended next steps for further analysis

Focus on providing actionable insights that would be valuable to business stakeholders."""

        return prompt

    def generate_visualization_prompt(
        self,
        dataset_context: DatasetContext,
        chart_request: str,
        analysis_context: Optional[str] = None
    ) -> str:
        """
        Generate a prompt focused on creating specific visualizations

        Args:
            dataset_context: Dataset information
            chart_request: Specific chart or visualization request
            analysis_context: Additional context about the analysis goal
        """

        dataset_info = self._format_dataset_context(dataset_context)

        context_section = ""
        if analysis_context:
            context_section = f"\n**ANALYSIS CONTEXT:**\n{analysis_context}\n"

        prompt = f"""**VISUALIZATION REQUEST:** "{chart_request}"

**DATASET CONTEXT:**
{dataset_info}
{context_section}

**VISUALIZATION TASK:**
As a professional data visualization expert, create compelling and statistically sound visualizations. Consider:

1. **Chart Selection**: Choose the most appropriate chart type for the data and message
2. **Statistical Accuracy**: Ensure visualizations don't mislead or misrepresent data
3. **Design Principles**: Follow best practices for color, layout, and accessibility
4. **Interactivity**: Leverage plotly's interactive features appropriately
5. **Statistical Annotations**: Add relevant statistical information (means, confidence intervals, etc.)

**TECHNICAL REQUIREMENTS:**
- Use plotly.express or plotly.graph_objects exclusively
- Include proper titles, axis labels, and legends
- Add hover information and tooltips
- Use appropriate color schemes and styling
- Handle edge cases (missing data, outliers)
- Make charts responsive and professional-looking

**STATISTICAL ENHANCEMENTS:**
- Add trend lines or regression lines where appropriate
- Include confidence bands for predictions
- Show distribution shapes (histograms, box plots) when relevant
- Highlight statistical significance in comparisons
- Add reference lines (means, medians, benchmarks)

**OUTPUT FORMAT:**
Provide:
1. **Visualization Strategy** - Why this chart type and approach
2. **Statistical Considerations** - What to highlight or be careful about
3. **Complete Python Code** - Ready-to-execute plotly code
4. **Interpretation Guide** - How to read and understand the chart
5. **Alternative Views** - Suggest 2-3 other ways to visualize the same data

Focus on creating publication-quality visualizations that communicate insights clearly."""

        return prompt

    def generate_dataset_joining_prompt(
        self,
        primary_dataset: DatasetContext,
        datasets_to_join: List[DatasetContext],
        join_objective: str,
        user_analysis_goal: str
    ) -> str:
        """
        Generate a prompt for intelligent dataset joining and analysis

        Args:
            primary_dataset: Main dataset to join to
            datasets_to_join: List of datasets to consider joining
            join_objective: What the user wants to achieve with joining
            user_analysis_goal: Overall analysis objective
        """

        primary_info = self._format_dataset_context(primary_dataset)

        datasets_info = "**DATASETS AVAILABLE FOR JOINING:**\n"
        for i, ds in enumerate(datasets_to_join, 1):
            datasets_info += f"\n**Dataset {i}: {ds.name}**\n"
            datasets_info += f"Description: {ds.description}\n"
            datasets_info += f"Rows: {ds.row_count:,}\n"
            datasets_info += f"Columns: {[col['name'] for col in ds.columns]}\n"
            datasets_info += f"Key columns: {[col['name'] for col in ds.columns if col.get('is_key', False)]}\n"

        prompt = f"""**ANALYSIS GOAL:** "{user_analysis_goal}"
**JOIN OBJECTIVE:** "{join_objective}"

**PRIMARY DATASET:**
{primary_info}

{datasets_info}

**DATASET INTEGRATION TASK:**
As a professional statistician with expertise in data integration, design and execute a comprehensive dataset joining strategy. Consider:

1. **Join Strategy Analysis**:
   - Identify potential join keys between datasets
   - Determine optimal join types (inner, left, outer, etc.)
   - Assess data quality and completeness for joining
   - Consider temporal alignment for time-series data

2. **Statistical Considerations**:
   - Evaluate impact of different join strategies on sample size
   - Consider selection bias introduced by joins
   - Plan for handling missing values post-join
   - Assess whether joins will introduce multicollinearity

3. **Data Quality Assessment**:
   - Check for duplicate keys and many-to-many relationships
   - Validate referential integrity between datasets
   - Identify potential data quality issues that could affect joins
   - Plan data cleaning steps before joining

**TECHNICAL REQUIREMENTS:**
- Use pandas merge, join, or concat operations appropriately
- Handle different data types and formats in join keys
- Implement proper error handling for failed joins
- Create validation checks for join results
- Generate summary statistics before and after joins

**DELIVERABLES:**
Please provide:

1. **Join Strategy Report**:
   - Recommended join approach and rationale
   - Identified join keys and their quality assessment
   - Expected impact on data size and completeness
   - Potential risks and mitigation strategies

2. **Complete Implementation Code**:
   - Data preparation and cleaning steps
   - Join operations with proper error handling
   - Data validation and quality checks
   - Summary statistics and join diagnostics

3. **Joined Dataset Analysis**:
   - Descriptive statistics of the combined dataset
   - Key insights enabled by the data integration
   - Visualization of relationships across datasets
   - Recommendations for further analysis

4. **Quality Assessment**:
   - Join success rate and data loss analysis
   - Identification of potential data quality issues
   - Recommendations for data cleaning and validation

Focus on creating a robust, well-documented data integration that enables meaningful analysis while maintaining data integrity."""

        return prompt

    def generate_statistical_testing_prompt(
        self,
        dataset_context: DatasetContext,
        hypothesis: str,
        test_type: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for statistical hypothesis testing

        Args:
            dataset_context: Dataset information
            hypothesis: Hypothesis to test
            test_type: Specific statistical test to use (optional)
        """

        dataset_info = self._format_dataset_context(dataset_context)

        test_guidance = ""
        if test_type:
            test_guidance = f"\n**REQUESTED TEST TYPE:** {test_type}\n"

        prompt = f"""**HYPOTHESIS TO TEST:** "{hypothesis}"
{test_guidance}
**DATASET CONTEXT:**
{dataset_info}

**STATISTICAL TESTING TASK:**
As a professional statistician, design and execute a rigorous hypothesis testing procedure. Apply your expertise in:

1. **Hypothesis Formulation**:
   - Clearly state null and alternative hypotheses (H₀ and H₁)
   - Define the parameter of interest and effect size
   - Consider one-tailed vs two-tailed testing
   - Set appropriate significance level (α)

2. **Test Selection & Assumptions**:
   - Choose the most appropriate statistical test
   - Verify test assumptions (normality, independence, homoscedasticity)
   - Consider non-parametric alternatives if assumptions are violated
   - Plan for multiple testing corrections if needed

3. **Sample Size & Power Analysis**:
   - Assess whether sample size is adequate
   - Calculate statistical power for the chosen test
   - Consider effect size and practical significance
   - Recommend minimum sample size if current is insufficient

4. **Data Preparation**:
   - Handle missing values appropriately for the test
   - Check for and address outliers
   - Transform data if required by test assumptions
   - Create appropriate groupings or categories

**TECHNICAL IMPLEMENTATION:**
- Use scipy.stats for statistical tests
- Use pandas for data preparation and manipulation
- Create plotly visualizations to support findings
- Include proper diagnostic plots (Q-Q plots, residual plots, etc.)
- Calculate effect sizes and confidence intervals

**DELIVERABLES:**
Provide a complete statistical analysis including:

1. **Experimental Design**:
   - Hypothesis statements (H₀ and H₁)
   - Test selection rationale and assumptions check
   - Significance level and power analysis
   - Sample size adequacy assessment

2. **Statistical Analysis Code**:
   - Data preparation and assumption checking
   - Complete test implementation with scipy.stats
   - Effect size calculations
   - Confidence interval estimation

3. **Results Interpretation**:
   - Test statistic and p-value interpretation
   - Effect size and practical significance
   - Confidence intervals and their meaning
   - Statistical vs practical significance discussion

4. **Visualizations**:
   - Distribution plots and assumption checks
   - Test results visualization
   - Effect size illustration
   - Confidence interval plots

5. **Professional Report**:
   - Executive summary of findings
   - Methodology explanation for non-statisticians
   - Limitations and caveats
   - Recommendations for follow-up analyses

Apply the rigor expected in peer-reviewed statistical research while making findings accessible to business stakeholders."""

        return prompt

    def _format_dataset_context(self, dataset_context: DatasetContext) -> str:
        """Format dataset context into a readable string"""

        # Format column information
        columns_info = "**Columns:**\n"
        for col in dataset_context.columns:
            col_name = col['name']
            col_type = col.get('type', 'unknown')
            col_desc = col.get('description', 'No description')
            missing_pct = (dataset_context.missing_values.get(col_name, 0) / dataset_context.row_count * 100) if dataset_context.row_count > 0 else 0
            unique_count = dataset_context.unique_counts.get(col_name, 'unknown')

            columns_info += f"- **{col_name}** ({col_type}): {col_desc}\n"
            columns_info += f"  Missing: {missing_pct:.1f}% | Unique values: {unique_count}\n"

        # Format sample data
        sample_info = "**Sample Data:**\n"
        for i, row in enumerate(dataset_context.sample_data[:5], 1):
            sample_info += f"Row {i}: {row}\n"

        # Build complete context
        context = f"""**Dataset: {dataset_context.name}**
Description: {dataset_context.description}
Total Rows: {dataset_context.row_count:,}
Total Columns: {len(dataset_context.columns)}

{columns_info}

{sample_info}

**Data Quality Summary:**
- Missing Values: {sum(dataset_context.missing_values.values()):,} total across all columns
- Data Types: {dict(dataset_context.data_types)}
"""
        return context

    def create_dataset_context_from_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = "Dataset for analysis"
    ) -> DatasetContext:
        """
        Create a DatasetContext from a pandas DataFrame

        Args:
            df: Pandas DataFrame
            name: Dataset name
            description: Dataset description

        Returns:
            DatasetContext object
        """

        # Get column information
        columns = []
        for col in df.columns:
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'description': f"Column {col} with {df[col].dtype} data type"
            }
            columns.append(col_info)

        # Get sample data (first 10 rows)
        sample_data = df.head(10).to_dict('records')

        # Get data quality information
        missing_values = df.isnull().sum().to_dict()
        unique_counts = df.nunique().to_dict()
        data_types = df.dtypes.astype(str).to_dict()

        return DatasetContext(
            name=name,
            description=description,
            columns=columns,
            sample_data=sample_data,
            row_count=len(df),
            data_types=data_types,
            missing_values=missing_values,
            unique_counts=unique_counts
        )

    def generate_comprehensive_eda_prompt(
        self,
        dataset_context: DatasetContext,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Generate a prompt for comprehensive Exploratory Data Analysis (EDA)

        Args:
            dataset_context: Dataset information
            focus_areas: Specific areas to focus on (e.g., 'distributions', 'correlations', 'outliers')
        """

        dataset_info = self._format_dataset_context(dataset_context)

        focus_section = ""
        if focus_areas:
            focus_section = f"\n**FOCUS AREAS:** {', '.join(focus_areas)}\n"

        prompt = f"""**COMPREHENSIVE EXPLORATORY DATA ANALYSIS REQUEST**

**DATASET CONTEXT:**
{dataset_info}
{focus_section}

**EDA TASK:**
As a professional statistician, conduct a thorough exploratory data analysis that would meet the standards of a senior data scientist at a Fortune 500 company. Your analysis should be:

1. **Systematic & Comprehensive**:
   - Follow a structured EDA methodology
   - Cover all major aspects of data exploration
   - Identify patterns, anomalies, and relationships
   - Assess data quality and reliability

2. **Statistically Sound**:
   - Use appropriate statistical measures and tests
   - Consider distributional assumptions
   - Apply robust methods for outlier detection
   - Calculate meaningful effect sizes

3. **Business-Focused**:
   - Translate statistical findings into business insights
   - Identify actionable opportunities
   - Highlight potential risks and limitations
   - Suggest data-driven recommendations

**ANALYSIS FRAMEWORK:**

**Phase 1: Data Quality Assessment**
- Missing value analysis and patterns
- Duplicate record detection
- Data type validation and consistency
- Outlier detection using statistical methods
- Data integrity and referential consistency

**Phase 2: Univariate Analysis**
- Distribution analysis for all variables
- Central tendency and variability measures
- Skewness, kurtosis, and normality testing
- Categorical variable frequency analysis
- Temporal patterns for date/time variables

**Phase 3: Bivariate Analysis**
- Correlation analysis (Pearson, Spearman, Kendall)
- Cross-tabulation for categorical variables
- Statistical significance testing for relationships
- Scatter plot analysis with trend identification
- Association measures for mixed variable types

**Phase 4: Multivariate Analysis**
- Principal Component Analysis (PCA) if appropriate
- Cluster analysis for pattern identification
- Multiple correlation and partial correlations
- Interaction effect exploration
- Dimensionality assessment

**VISUALIZATION REQUIREMENTS:**
Create a comprehensive set of plotly visualizations:
- Distribution plots (histograms, box plots, violin plots)
- Correlation heatmaps and scatter plot matrices
- Time series plots for temporal data
- Interactive dashboards for exploration
- Statistical diagnostic plots

**DELIVERABLES:**
Provide a complete EDA report with:

1. **Executive Summary** (2-3 paragraphs)
   - Key findings and insights
   - Data quality assessment
   - Business implications

2. **Technical Analysis** (Complete Python code)
   - All statistical calculations
   - Comprehensive visualizations
   - Assumption testing and validation

3. **Detailed Findings** (By analysis phase)
   - Data quality issues and recommendations
   - Distribution characteristics and implications
   - Relationship strength and significance
   - Pattern identification and interpretation

4. **Business Insights**
   - Actionable recommendations
   - Risk identification
   - Opportunity assessment
   - Next steps for deeper analysis

5. **Technical Appendix**
   - Statistical test results
   - Assumption checking
   - Methodology justification
   - Limitations and caveats

Focus on providing insights that would drive business decisions and guide further analytical work."""

        return prompt