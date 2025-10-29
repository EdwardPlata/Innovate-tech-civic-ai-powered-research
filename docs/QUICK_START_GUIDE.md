# Quick Start Guide - Dataset Chat Feature

## ✅ Backend is Already Running!

Great news! Your backend API is already running and working perfectly. I can see from the logs that it's serving requests on `http://localhost:8080`.

## 🚀 How to Access the Dataset Chat Feature

### Step 1: Start the Frontend (Streamlit)

The issue you're experiencing is that the frontend needs to be run in the same virtual environment as the backend. Here's how:

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Start the Streamlit frontend
cd frontend
streamlit run app.py
```

### Step 2: Use the Dataset Chat Feature

1. **Navigate to "Data Sample" page** (from the sidebar menu)

2. **Select a dataset first:**
   - Go to "Dashboard" or "Dataset Explorer"
   - Choose any dataset (like the 311 dataset you mentioned)
   - This will set `st.session_state.selected_dataset`

3. **Load the dataset into memory:**
   - On the "Data Sample" page, you'll see "📥 Load Dataset" section
   - Choose sample size (default: 500 rows)
   - Click "🚀 Load Dataset for AI Chat"

4. **Start chatting with your data:**
   - Use the chat interface on the right side
   - Try quick action buttons:
     - "📈 Show Summary"
     - "🔍 Find Patterns"
     - "⚠️ Check Quality"
   - Or ask custom questions like:
     - "What are the main patterns in this data?"
     - "Show me statistics for the [column_name] column"
     - "Are there any outliers?"

## 🔧 Troubleshooting

If you still see "Backend API is not running":

### Option 1: Refresh the Status
- Click the "🔄 Refresh Status" button in the sidebar
- The backend should be detected as online

### Option 2: Force Backend Detection
```bash
# Test if backend is responding
curl http://localhost:8080/api/health

# Should return:
# {"status":"healthy","services":{"scout_discovery":true,"cache_manager":true,"ai_functionality":false}...}
```

### Option 3: Restart Streamlit with Proper Environment
```bash
# Make sure you're in the project root
cd /home/edward/code/QLT_Workshop

# Activate virtual environment
source .venv/bin/activate

# Start frontend
cd frontend
streamlit run app.py
```

## 📊 What You Can Do with Dataset Chat

### Load Any Dataset:
- 311 Service Requests
- Housing Data
- Transportation Data
- Health Inspections
- Any NYC Open Data

### Ask Natural Language Questions:
```
"What are the most common complaint types?"
"Show me the distribution of response times"
"Are there seasonal patterns in the data?"
"What boroughs have the most issues?"
"Find any outliers or unusual patterns"
```

### Get AI Insights:
- Specific analysis of your loaded data
- Statistical summaries
- Pattern detection
- Data quality assessment
- Python code suggestions

## 🎯 Example Workflow

1. **Dashboard** → Select "311 Service Requests" dataset
2. **Data Sample** → Load 500 rows into memory
3. **Chat**: "What are the most common complaint types in this data?"
4. **AI Response**: Analyzes your specific 500 rows and shows actual complaint type frequencies
5. **Follow-up**: "Show me the response time statistics"
6. **Continue** the conversation with more specific questions

## 📝 Key Features Working:

✅ **Backend API**: Running on port 8080
✅ **Dataset Loading**: Into session memory
✅ **Natural Language Chat**: With your specific data
✅ **AI Analysis**: Using actual data values
✅ **Multiple AI Providers**: OpenAI, OpenRouter, NVIDIA support
✅ **No Page Refreshes**: Seamless chat experience

The system is ready to use! Just make sure to run the frontend from the virtual environment.