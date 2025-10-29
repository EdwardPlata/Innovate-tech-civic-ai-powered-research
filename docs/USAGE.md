# Scout Data Discovery Platform - Usage Guide

## 🚀 Getting Started

### Step 1: Launch the Platform

```bash
cd frontend
python run_app.py
```

Open your browser to: http://localhost:8501

### Step 2: Start the Backend

When you first open the web app:

1. **You'll see**: "⚠️ Backend API is not running"
2. **Click**: "🚀 Start Backend Server"
3. **Wait**: ~15-30 seconds for the server to initialize
4. **Success**: "✅ Connected to Scout API" will appear

## 🔍 Using the Platform

### Dashboard
- View **top 10 recently updated datasets** from NYC Open Data
- See **key metrics**: total datasets, downloads, categories
- **Category distribution** pie chart
- **Select datasets** for detailed analysis

### Dataset Explorer
- **Search** datasets using keywords (e.g., "311", "health", "transportation")
- **Filter** by category, minimum downloads
- **Sort** by name, popularity, or update date
- **Preview** dataset information before selection

### Quality Assessment
1. **Select a dataset** from Dashboard or Explorer
2. **Click** "🔍 Assess Quality"
3. **View results**:
   - Overall score (0-100) and letter grade
   - 5-dimension breakdown (completeness, consistency, accuracy, timeliness, usability)
   - Actionable insights and recommendations

### Relationship Mapping
1. **Select a dataset** as your starting point
2. **Adjust settings**:
   - Similarity threshold (0.1-0.9)
   - Max related datasets (5-50)
3. **Click** "🔍 Find Relationships"
4. **Explore**:
   - Network visualization showing connections
   - Table of related datasets with similarity scores
   - Network statistics and metrics

### Data Sample Viewer
1. **Select a dataset**
2. **Choose sample size** (10-5000 rows)
3. **Click** "📥 Load Sample Data"
4. **Review**:
   - Column information and data types
   - Sample data preview
   - Download CSV for further analysis

## 🎯 Key Features

### Smart Dataset Discovery
- **Multi-term search**: Use multiple keywords separated by commas
- **Category filtering**: Focus on specific data domains
- **Popularity sorting**: Find the most used datasets
- **Real-time results**: Live search as you type

### AI-Powered Quality Assessment
- **5-Dimensional Scoring**: Comprehensive quality evaluation
- **Visual Dashboards**: Easy-to-understand quality metrics
- **Actionable Insights**: Specific recommendations for data improvement
- **Grade System**: A-F grades for quick quality assessment

### Advanced Relationship Analysis
- **Content Similarity**: Analyzes descriptions and metadata
- **Structural Similarity**: Compares column names and data types
- **Usage Patterns**: Considers download patterns and update frequency
- **Network Visualization**: Interactive graphs showing dataset connections

### User-Friendly Interface
- **Integrated Backend**: No manual server management
- **Real-time Status**: Live backend monitoring
- **Progressive Loading**: Large datasets loaded efficiently
- **Export Options**: Download results for further analysis

## 🔧 Backend Management

### Status Monitoring
The sidebar shows real-time backend status:
- **🟢 Online / 🔴 Offline**: Current server status
- **✅ Pass / ❌ Fail**: Health check results
- **🔄 Active / ⏸️ Inactive**: Process status

### Controls
- **🚀 Start Backend**: Launch the API server
- **🛑 Stop Backend**: Shut down the API server
- **🔄 Refresh Status**: Update status information

### Troubleshooting
If backend fails to start:
1. Check port 8000 isn't in use
2. Ensure all dependencies are installed
3. Manually start: `cd backend && uvicorn main:app --reload`

## 📊 Use Case Examples

### Finding Related 311 Datasets
1. **Search**: "311 service requests"
2. **Select**: Most popular 311 dataset
3. **Assess Quality**: Check data completeness
4. **Find Relationships**: Discover related city service datasets
5. **Sample Data**: Preview data structure for integration

### Data Quality Assessment Workflow
1. **Search**: Your domain of interest
2. **Compare**: Multiple datasets in search results
3. **Assess**: Quality scores for top candidates
4. **Select**: Highest quality datasets for your project

### Cross-Domain Discovery
1. **Start**: With a known dataset in your domain
2. **Map Relationships**: Find connected datasets in other domains
3. **Explore Network**: Understand data ecosystem connections
4. **Identify Opportunities**: Find unexpected data integration possibilities

## 🚨 Common Issues

### "Backend not responding"
- Wait 30 seconds and try again
- Check sidebar for backend status
- Restart backend using UI controls

### "No datasets found"
- Try different search terms
- Check spelling and use synonyms
- Use broader search terms (e.g., "health" instead of "healthcare statistics")

### "Quality assessment failed"
- Dataset might be empty or inaccessible
- Try a different, smaller dataset
- Check network connectivity

### "Network visualization empty"
- Lower similarity threshold (try 0.2 or 0.1)
- Increase max related datasets
- Ensure you have enough datasets in your search results

## 💡 Pro Tips

1. **Start Small**: Begin with popular datasets (high download counts)
2. **Use Keywords**: Try domain-specific terms like "311", "DOT", "DOH"
3. **Check Dates**: Recent updates often indicate actively maintained datasets
4. **Explore Categories**: Browse different city departments' data
5. **Export Results**: Download sample data for detailed analysis
6. **Network Analysis**: Use relationship mapping to discover hidden connections

## 🎯 Best Practices

### For Data Discovery
- Use multiple search terms for comprehensive results
- Filter by category to focus your search
- Sort by popularity to find proven datasets

### For Quality Assessment
- Always assess quality before using data in production
- Look for Grade A or B datasets for reliable results
- Pay attention to completeness scores for critical analysis

### For Relationship Mapping
- Start with similarity threshold of 0.3
- Gradually lower threshold to find more connections
- Use network visualization to understand data ecosystem

---

**Need Help?** The platform includes built-in guidance and real-time status information. Most issues resolve with a backend restart or trying different search parameters.