# Codebase Agent - AI-Powered Code Analysis Documentation

## 📋 Overview

The `codebase_agent.py` module provides an intelligent codebase analysis system that uses AI to understand, analyze, and answer questions about code repositories. It combines static code analysis with AI-powered insights to deliver comprehensive codebase intelligence.

## 🎯 Purpose

- **Primary Role**: AI-powered codebase understanding and analysis
- **Key Responsibility**: Transform code repositories into queryable knowledge bases
- **Core Function**: Intelligent code chunking, analysis, and Q&A capabilities
- **Integration Point**: Bridge between static code analysis and AI-powered insights

## 🏗️ Architecture

```python
CodebaseAgent
├── Code Analysis Pipeline
│   ├── File Discovery (multi-language support)
│   ├── Intelligent Chunking (AST-based parsing)
│   ├── Metadata Extraction (functions, classes, docs)
│   └── Dependency Mapping (cross-references)
├── AI-Powered Analysis
│   ├── Architecture Assessment (design patterns)
│   ├── Quality Evaluation (maintainability scoring)
│   ├── Security Analysis (vulnerability detection)
│   └── Interactive Q&A (natural language queries)
├── Caching & Storage
│   ├── Chunk Persistence (incremental updates)
│   ├── Analysis Caching (performance optimization)
│   ├── Index Management (fast retrieval)
│   └── Version Tracking (change detection)
└── Multi-Language Support
    ├── Python (AST parsing)
    ├── JavaScript/TypeScript (pattern matching)
    ├── Configuration Files (JSON, YAML)
    └── Documentation (Markdown, text)
```

## 📊 Data Models

### CodeChunk

**Fundamental unit of code analysis**

```python
@dataclass
class CodeChunk:
    """
    Represents a semantically meaningful piece of code with rich metadata
    
    Each chunk captures a logical unit of code (function, class, module section)
    along with contextual information for AI analysis.
    """
    
    # Core Identity
    id: str                           # Unique identifier (file:name:line)
    file_path: str                    # Absolute path to source file
    chunk_type: CodeChunkType         # Type of code construct
    name: str                         # Human-readable name
    
    # Content & Location
    content: str                      # Actual code content
    start_line: int                   # Starting line number
    end_line: int                     # Ending line number
    
    # Relationships & Context
    dependencies: List[str] = None    # Referenced modules/functions
    docstring: Optional[str] = None   # Documentation string
    
    # Analysis Metrics
    complexity_score: float = 0.0     # Calculated complexity (0-1)
    metadata: Dict[str, Any] = None   # Language-specific metadata
    
    def __post_init__(self):
        """Initialize collections if None"""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

# Usage Examples:

# Python function chunk
function_chunk = CodeChunk(
    id="analytics.py:calculate_quality_score:45",
    file_path="/src/analytics.py", 
    chunk_type=CodeChunkType.FUNCTION,
    name="calculate_quality_score",
    content="""def calculate_quality_score(data):
    \"\"\"Calculate overall data quality score.\"\"\"
    completeness = check_completeness(data)
    accuracy = check_accuracy(data)
    return (completeness + accuracy) / 2""",
    start_line=45,
    end_line=50,
    docstring="Calculate overall data quality score.",
    dependencies=["check_completeness", "check_accuracy"],
    complexity_score=0.3,
    metadata={
        "args": ["data"],
        "return_type": "float",
        "calls": ["check_completeness", "check_accuracy"]
    }
)

# Python class chunk
class_chunk = CodeChunk(
    id="models.py:DataAnalyzer:12",
    file_path="/src/models.py",
    chunk_type=CodeChunkType.CLASS,
    name="DataAnalyzer",
    content="""class DataAnalyzer:
    \"\"\"Main class for data analysis operations.\"\"\"
    
    def __init__(self, config):
        self.config = config
        
    def analyze(self, dataset):
        # Analysis implementation
        pass""",
    start_line=12,
    end_line=22,
    docstring="Main class for data analysis operations.",
    dependencies=["config"],
    complexity_score=0.4,
    metadata={
        "bases": [],
        "methods": ["__init__", "analyze"],
        "attributes": ["config"]
    }
)
```

### CodeChunkType Enumeration

**Categories of code constructs for analysis**

```python
class CodeChunkType(Enum):
    """
    Types of code chunks for structured analysis
    """
    
    FUNCTION = "function"
    """
    Function definitions and methods
    - Standalone functions
    - Class methods
    - Lambda expressions
    - Async functions
    """
    
    CLASS = "class"
    """
    Class definitions and interfaces
    - Class declarations
    - Interface definitions
    - Abstract base classes
    - Data classes/structs
    """
    
    MODULE = "module"
    """
    Module-level code and sections
    - Module initialization
    - Global variables
    - Module constants
    - Top-level statements
    """
    
    IMPORT = "import"
    """
    Import statements and dependencies
    - Import declarations
    - From imports
    - Dynamic imports
    - Module aliases
    """
    
    COMMENT = "comment"
    """
    Documentation and comments
    - Block comments
    - Inline documentation
    - Docstrings
    - README sections
    """
    
    CONFIG = "config"
    """
    Configuration and data files
    - JSON configuration
    - YAML settings
    - Environment files
    - Data schemas
    """
```

## 🧠 Core Classes

### CodebaseAgent

**Main orchestrator for codebase analysis and AI-powered insights**

```python
class CodebaseAgent:
    """
    AI-powered codebase analysis and Q&A agent
    
    Features:
    - Intelligent multi-language code chunking
    - AST-based parsing for precise analysis
    - AI-powered architecture and quality assessment
    - Interactive Q&A about codebase structure and patterns
    - Comprehensive caching for performance optimization
    - Incremental analysis for large codebases
    """
    
    def __init__(
        self,
        codebase_path: str,
        ai_analyst: DataAnalyst,
        cache_dir: str = "./codebase_cache",
        supported_extensions: List[str] = None
    ):
        """
        Initialize CodebaseAgent with configuration
        
        Args:
            codebase_path: Path to the codebase root directory
                - Can be relative or absolute path
                - Should point to repository root for best results
                - Automatically traverses subdirectories
            
            ai_analyst: DataAnalyst instance for AI operations
                - Provides AI-powered analysis capabilities
                - Handles multiple provider fallbacks
                - Manages caching and performance optimization
            
            cache_dir: Directory for caching chunked code and analysis
                - Stores processed chunks for fast reloading
                - Includes analysis results and metadata
                - Automatically cleaned based on file changes
            
            supported_extensions: File extensions to analyze
                - Defaults to comprehensive language support
                - Can be customized for specific tech stacks
                - Includes code, config, and documentation files
        
        Supported Languages & Files:
        - Python: .py (full AST parsing)
        - JavaScript/TypeScript: .js, .ts, .jsx, .tsx
        - Documentation: .md, .txt, .rst
        - Configuration: .json, .yml, .yaml
        - Other: .go, .java, .cpp, .c, .h, .cs, .php, .rb, .rs
        """
        
        self.codebase_path = Path(codebase_path)
        self.ai_analyst = ai_analyst
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=str(self.cache_dir),
            enable_semantic=True
        )
        
        # Configure supported file types
        self.supported_extensions = supported_extensions or [
            # Programming languages
            '.py', '.js', '.ts', '.jsx', '.tsx',      # Web & Python
            '.go', '.java', '.cpp', '.c', '.h',       # Systems languages  
            '.cs', '.php', '.rb', '.rs',              # Other languages
            # Configuration & docs
            '.json', '.yml', '.yaml', '.toml',        # Config files
            '.md', '.txt', '.rst', '.sql',            # Documentation
        ]
        
        # Analysis state
        self.chunks: List[CodeChunk] = []
        self.file_index: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.codebase_summary: Optional[Dict[str, Any]] = None
        
        logger.info(f"CodebaseAgent initialized for: {self.codebase_path}")
```

## 🔧 Core Methods

### chunk_codebase()

**Intelligent chunking of codebase into analyzable units**

```python
def chunk_codebase(self, progress_callback: Optional[callable] = None) -> int:
    """
    Intelligently chunk the codebase into semantically meaningful pieces
    
    Chunking Process:
    1. File Discovery: Scan for supported file types
    2. Change Detection: Check for modifications since last chunking
    3. Language-Specific Parsing: Use appropriate parser for each file type
    4. Metadata Extraction: Extract docstrings, dependencies, complexity
    5. Caching: Store chunks for fast future access
    6. Index Building: Create search indexes for efficient querying
    
    Args:
        progress_callback: Optional function for progress updates
            Signature: callback(message: str, progress: float)
            progress: 0.0 to 1.0 indicating completion percentage
    
    Returns:
        int: Number of chunks created
    
    Chunking Strategies by Language:
    
    Python (.py):
    - Uses AST parsing for precise function/class extraction
    - Captures docstrings, arguments, and call relationships
    - Calculates cyclomatic complexity
    - Identifies imports and dependencies
    
    JavaScript/TypeScript (.js, .ts, .jsx, .tsx):
    - Pattern-based chunking for functions and classes
    - Detects arrow functions and modern syntax
    - Captures JSDoc comments
    - Identifies ES6 imports and exports
    
    Configuration (.json, .yml, .yaml):
    - Single chunk per file with structured parsing
    - Extracts configuration schema and values
    - Identifies environment-specific settings
    
    Documentation (.md, .txt, .rst):
    - Section-based chunking by headers
    - Preserves document structure
    - Extracts code examples
    """
    
    logger.info("Starting intelligent codebase chunking...")
    
    if progress_callback:
        progress_callback("🔍 Scanning codebase files...", 0.1)
    
    # Discover all relevant files
    files_to_analyze = self._find_code_files()
    
    if progress_callback:
        progress_callback(f"📁 Found {len(files_to_analyze)} files to analyze", 0.2)
    
    # Check for cached chunks (based on file modification times)
    cache_key = self._get_codebase_cache_key(files_to_analyze)
    cached_chunks = self._load_cached_chunks(cache_key)
    
    if cached_chunks:
        self.chunks = cached_chunks
        logger.info(f"♻️ Loaded {len(self.chunks)} cached chunks")
        if progress_callback:
            progress_callback("✅ Loaded cached chunks", 1.0)
        return len(self.chunks)
    
    # Process files with language-specific chunking
    self.chunks = []
    total_files = len(files_to_analyze)
    
    for i, file_path in enumerate(files_to_analyze):
        if progress_callback:
            progress = 0.2 + (0.7 * (i / total_files))
            progress_callback(f"🔧 Processing {file_path.name}...", progress)
        
        try:
            file_chunks = self._chunk_file(file_path)
            self.chunks.extend(file_chunks)
            
            # Log progress for large files
            if len(file_chunks) > 10:
                logger.info(f"Chunked {file_path.name}: {len(file_chunks)} chunks")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to chunk {file_path}: {e}")
            # Continue with other files instead of failing completely
    
    # Cache the processed chunks
    self._cache_chunks(cache_key, self.chunks)
    
    # Build search indexes for efficient querying
    if progress_callback:
        progress_callback("🗂️ Building search indexes...", 0.9)
    
    self._build_indexes()
    
    logger.info(f"✅ Codebase chunking complete: {len(self.chunks)} chunks created")
    
    if progress_callback:
        progress_callback("🎉 Chunking complete!", 1.0)
    
    return len(self.chunks)

# Usage example with progress tracking:

def chunking_progress(message: str, progress: float):
    """Progress callback for chunking operation"""
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"\r{message} [{bar}] {progress:.1%}", end='', flush=True)

# Chunk the codebase with progress updates
chunk_count = codebase_agent.chunk_codebase(progress_callback=chunking_progress)
print(f"\nCreated {chunk_count} code chunks for analysis")
```

### analyze_codebase()

**AI-powered comprehensive codebase analysis**

```python
def analyze_codebase(self, analysis_focus: str = "overview") -> Dict[str, Any]:
    """
    Perform AI-powered comprehensive codebase analysis
    
    Analysis Types:
    
    overview: General codebase understanding
    - Architecture patterns and structure
    - Technology stack identification
    - Code organization assessment
    - Key components and their roles
    - Dependencies and relationships
    - Complexity and maintainability overview
    
    architecture: Deep architectural analysis
    - Design patterns identification
    - Module interaction patterns
    - Data flow analysis
    - Separation of concerns evaluation
    - Scalability assessment
    - Technical debt identification
    
    quality: Code quality and maintainability assessment
    - Code style consistency
    - Documentation quality
    - Test coverage analysis
    - Error handling patterns
    - Performance considerations
    - Maintainability scoring
    
    security: Security-focused analysis
    - Input validation patterns
    - Authentication/authorization mechanisms
    - Data protection practices
    - Dependency security assessment
    - Common vulnerability patterns
    - Security best practices compliance
    
    Args:
        analysis_focus: Type of analysis to perform
    
    Returns:
        Dict containing:
        - analysis: AI-generated analysis text
        - analysis_type: Type of analysis performed
        - findings: Structured findings and recommendations
        - metrics: Quantitative metrics where applicable
        - timestamp: When analysis was performed
        - codebase_stats: Basic statistics about the codebase
    """
    
    if not self.chunks:
        raise ValueError("Codebase not chunked. Call chunk_codebase() first.")
    
    logger.info(f"🧠 Starting AI codebase analysis: {analysis_focus}")
    
    # Check for cached analysis results
    cache_key = f"codebase_analysis_{analysis_focus}_{len(self.chunks)}"
    cached_analysis = self.cache_manager.get_cached_context(cache_key)
    
    if cached_analysis:
        logger.info("♻️ Using cached codebase analysis")
        return cached_analysis
    
    # Build comprehensive context for AI analysis
    context = self._build_codebase_context()
    
    # Create analysis prompts for different focus areas
    analysis_prompts = {
        "overview": f"""
        Analyze this codebase and provide a comprehensive overview:

        {context}

        Please provide a detailed analysis covering:

        🏗️ **Architecture Overview**
        - High-level structure and organizational patterns
        - Main architectural components and their relationships
        - Design patterns and architectural styles used

        🔧 **Key Components**
        - Primary modules, classes, and their purposes
        - Core functionality and business logic
        - Entry points and main workflows

        📚 **Technology Stack**
        - Programming languages and versions
        - Frameworks, libraries, and dependencies
        - Development tools and build systems

        📁 **Code Organization**
        - Directory structure and naming conventions
        - Module organization and separation of concerns
        - Configuration and documentation structure

        🔗 **Dependencies & Relationships**
        - External libraries and their usage
        - Internal module dependencies
        - Integration points and interfaces

        📊 **Complexity Assessment**
        - Overall complexity level (1-10 scale)
        - Most complex components
        - Maintainability considerations

        💡 **Recommendations**
        - Areas for improvement
        - Modernization opportunities
        - Best practices to implement

        Focus on actionable insights for developers new to this codebase.
        """,

        "architecture": f"""
        Perform a deep architectural analysis of this codebase:

        {context}

        Focus on:

        🎯 **Design Patterns**
        - Identify architectural and design patterns in use
        - Evaluate pattern implementation quality
        - Suggest pattern improvements or additions

        🔄 **Module Relationships**
        - Analyze how components interact and depend on each other
        - Identify coupling levels and cohesion
        - Map data and control flow between modules

        📈 **Data Flow**
        - Trace how data moves through the system
        - Identify data transformation points
        - Analyze data persistence and caching strategies

        🎭 **Separation of Concerns**
        - Evaluate how responsibilities are divided
        - Identify violations of single responsibility principle
        - Suggest refactoring opportunities

        🚀 **Scalability**
        - Assess the architecture's ability to scale
        - Identify potential bottlenecks
        - Suggest scalability improvements

        🛠️ **Maintainability**
        - Evaluate ease of modification and extension
        - Identify areas that resist change
        - Suggest architectural improvements

        🏗️ **Technical Debt**
        - Identify architectural anti-patterns
        - Suggest refactoring priorities
        - Estimate effort for improvements

        Provide specific recommendations for architectural improvements.
        """,

        "quality": f"""
        Assess the code quality and maintainability of this codebase:

        {context}

        Evaluate:

        🎨 **Code Style**
        - Consistency in naming conventions
        - Code formatting and organization
        - Adherence to language-specific conventions

        📖 **Documentation**
        - Quality and completeness of comments
        - Docstring coverage and quality
        - README and documentation files

        🧪 **Testing**
        - Test coverage and quality
        - Testing strategies and patterns
        - Test organization and maintainability

        ⚠️ **Error Handling**
        - Robustness and error management patterns
        - Exception handling strategies
        - Input validation and sanitization

        ⚡ **Performance**
        - Potential performance bottlenecks
        - Resource usage patterns
        - Optimization opportunities

        🔒 **Security**
        - Security best practices compliance
        - Potential vulnerability patterns
        - Data protection measures

        📊 **Maintainability Score**
        - Overall maintainability rating (1-10)
        - Factors affecting maintainability
        - Improvement recommendations

        Highlight specific files or components that need immediate attention.
        Provide actionable recommendations with priority levels.
        """,

        "security": f"""
        Perform a comprehensive security analysis of this codebase:

        {context}

        Examine:

        🛡️ **Input Validation**
        - How user input is validated and sanitized
        - Potential injection vulnerabilities
        - Data validation patterns and gaps

        🔐 **Authentication/Authorization**
        - Authentication mechanisms implemented
        - Authorization and access control patterns
        - Session management practices

        🔒 **Data Protection**
        - Sensitive data handling practices
        - Encryption and hashing usage
        - Data storage and transmission security

        📦 **Dependency Security**
        - Third-party library security assessment
        - Dependency update practices
        - Known vulnerability exposure

        ⚙️ **Configuration Security**
        - Secure configuration practices
        - Environment variable handling
        - Secret management approaches

        🎯 **Common Vulnerabilities**
        - OWASP Top 10 considerations
        - Language-specific vulnerability patterns
        - Framework-specific security issues

        🚨 **Security Recommendations**
        - Immediate security improvements needed
        - Security best practices to implement
        - Long-term security strategy suggestions

        Focus on identifying specific vulnerabilities and providing actionable mitigation strategies.
        Rate security posture on a scale of 1-10 with detailed justification.
        """
    }
    
    # Get analysis prompt for the specified focus
    prompt = analysis_prompts.get(analysis_focus, analysis_prompts["overview"])
    
    try:
        # Execute AI analysis
        analysis_result = self._run_ai_analysis(prompt, analysis_focus)
        
        # Enhance with codebase statistics
        analysis_result.update({
            "codebase_stats": self.get_codebase_stats(),
            "analysis_focus": analysis_focus,
            "chunk_count": len(self.chunks),
            "file_count": len(self.file_index)
        })
        
        # Cache the comprehensive analysis
        self.cache_manager.cache_context_analysis(cache_key, analysis_result)
        
        logger.info(f"✅ Codebase analysis completed: {analysis_focus}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ Codebase analysis failed: {e}")
        return {
            "error": str(e),
            "analysis_type": analysis_focus,
            "timestamp": self._get_timestamp(),
            "codebase_stats": self.get_codebase_stats() if self.chunks else {}
        }

# Usage examples:

# Basic overview analysis
overview = codebase_agent.analyze_codebase("overview")
print(f"📊 Codebase Overview:")
print(f"Files analyzed: {overview['file_count']}")
print(f"Chunks created: {overview['chunk_count']}")
print(f"\n{overview['analysis']}")

# Architecture deep-dive
architecture = codebase_agent.analyze_codebase("architecture")
print(f"🏗️ Architecture Analysis:")
print(architecture['analysis'])

# Security assessment
security = codebase_agent.analyze_codebase("security")
print(f"🔒 Security Analysis:")
print(security['analysis'])
```

### answer_codebase_question()

**Interactive AI-powered Q&A about the codebase**

```python
def answer_codebase_question(self, 
                           question: str, 
                           context_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Answer natural language questions about the codebase using AI
    
    Question Types Supported:
    
    Functional Questions:
    - "How does user authentication work?"
    - "Where is data validation implemented?"
    - "What design patterns are used here?"
    
    Location Questions:
    - "Where is the login functionality?"
    - "Which file contains the database models?"
    - "Where are the API endpoints defined?"
    
    Implementation Questions:
    - "How can I add a new feature to X?"
    - "What's the best way to extend Y?"
    - "How do I modify the Z behavior?"
    
    Quality Questions:
    - "What are the main technical debt areas?"
    - "Which components need refactoring?"
    - "How can performance be improved?"
    
    Args:
        question: Natural language question about the codebase
        context_files: Optional list of specific files to focus on
            - Narrows search scope for better relevance
            - Useful for file-specific questions
            - Can use partial path matching
    
    Returns:
        Dict containing:
        - answer: Direct answer to the question
        - relevant_files: Files referenced in the answer
        - code_references: Specific code sections mentioned
        - related_components: Other relevant parts of codebase
        - confidence: AI confidence in the answer (0-1)
        - search_scope: Files searched for the answer
    """
    
    if not self.chunks:
        raise ValueError("Codebase not chunked. Call chunk_codebase() first.")
    
    logger.info(f"🤔 Answering codebase question: {question[:100]}...")
    
    # Find chunks relevant to the question
    relevant_chunks = self._find_relevant_chunks(question, context_files)
    
    if not relevant_chunks:
        return {
            "answer": "No relevant code found for this question.",
            "question": question,
            "relevant_files": [],
            "search_scope": context_files or "entire_codebase",
            "confidence": 0.0
        }
    
    # Build focused context from relevant chunks
    context = self._build_question_context(relevant_chunks, question)
    
    # Create comprehensive Q&A prompt
    prompt = f"""
    Based on the following codebase information, please answer this question: "{question}"

    Codebase Context:
    {context}

    Please provide a comprehensive answer that includes:

    🎯 **Direct Answer**
    - Clear, specific answer to the question
    - Address the question directly and completely

    📍 **Code References**
    - Specific files, functions, and classes involved
    - Line numbers or code sections where relevant
    - Exact code snippets that demonstrate the answer

    🔧 **Implementation Details**
    - How the relevant code works
    - Key algorithms or patterns used
    - Important configuration or setup requirements

    🔗 **Related Components**
    - Other parts of the codebase that are related
    - Dependencies and interactions
    - Integration points and interfaces

    💡 **Practical Examples**
    - Code examples if helpful for understanding
    - Usage patterns or common scenarios
    - Best practices for working with this code

    📋 **Recommendations**
    - Suggestions for improvement (if applicable)
    - Best practices to follow
    - Common pitfalls to avoid

    Be specific and reference actual code elements when possible.
    If the question cannot be fully answered from the available code, say so clearly.
    """
    
    try:
        # Execute AI Q&A analysis
        response = self._run_ai_analysis(prompt, "question_answer")
        
        # Extract code references and metadata
        relevant_files = list(set(chunk.file_path for chunk in relevant_chunks))
        
        # Calculate confidence based on relevance and coverage
        confidence = min(1.0, len(relevant_chunks) / 10.0)  # Simple heuristic
        
        # Structure the response
        structured_response = {
            "answer": response.get("analysis", "Analysis not available"),
            "question": question,
            "relevant_files": relevant_files,
            "chunks_analyzed": len(relevant_chunks),
            "search_scope": context_files or "entire_codebase",
            "confidence": confidence,
            "timestamp": response.get("timestamp"),
            "provider": response.get("provider"),
            "cached": response.get("cached", False)
        }
        
        # Add code references if available
        code_references = []
        for chunk in relevant_chunks[:5]:  # Top 5 most relevant
            code_references.append({
                "file": chunk.file_path,
                "name": chunk.name,
                "type": chunk.chunk_type.value,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            })
        
        structured_response["code_references"] = code_references
        
        logger.info(f"✅ Question answered with {len(relevant_chunks)} relevant chunks")
        return structured_response
        
    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        return {
            "error": str(e),
            "question": question,
            "timestamp": self._get_timestamp(),
            "search_scope": context_files or "entire_codebase"
        }

# Usage examples:

# General functionality question
response = codebase_agent.answer_codebase_question(
    "How does user authentication work in this application?"
)
print(f"🔐 Authentication Implementation:")
print(f"Answer: {response['answer']}")
print(f"Files involved: {', '.join(response['relevant_files'])}")

# Specific implementation question
response = codebase_agent.answer_codebase_question(
    "Where is database connection configuration handled?",
    context_files=["config", "database", "models"]  # Focus search
)
print(f"💾 Database Configuration:")
print(f"Answer: {response['answer']}")

# Code quality question
response = codebase_agent.answer_codebase_question(
    "What are the main areas that need refactoring?"
)
print(f"🔧 Refactoring Opportunities:")
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.1%}")
```

### get_code_suggestions()

**AI-powered code improvement suggestions for specific files**

```python
def get_code_suggestions(self, 
                        file_path: str, 
                        improvement_type: str = "all") -> Dict[str, Any]:
    """
    Get AI-powered suggestions for improving specific files
    
    Improvement Types:
    
    all: Comprehensive analysis covering all aspects
    - Readability and code style
    - Performance optimizations
    - Security considerations
    - Maintainability improvements
    
    performance: Focus on performance optimizations
    - Algorithm efficiency improvements
    - Memory usage optimizations
    - I/O and database query optimization
    - Caching opportunities
    
    readability: Focus on code clarity and style
    - Variable and function naming
    - Code organization and structure
    - Documentation and comments
    - Simplification opportunities
    
    security: Focus on security vulnerabilities
    - Input validation issues
    - Authentication/authorization problems
    - Data protection concerns
    - Dependency security issues
    
    maintainability: Focus on long-term code health
    - Code duplication reduction
    - Modularity improvements
    - Testing and documentation
    - Extensibility enhancements
    
    Args:
        file_path: Path to the specific file to analyze
        improvement_type: Type of improvements to focus on
    
    Returns:
        Dict containing:
        - suggestions: List of specific improvement recommendations
        - priority_issues: High-priority problems to address
        - code_examples: Before/after code snippets
        - effort_estimate: Estimated effort for each suggestion
        - impact_assessment: Expected impact of improvements
    """
    
    # Find chunks for the specific file
    file_chunks = [chunk for chunk in self.chunks if chunk.file_path == file_path]
    
    if not file_chunks:
        return {
            "error": f"File not found in analyzed codebase: {file_path}",
            "file_path": file_path,
            "suggestion": "Ensure the file exists and was included in chunking"
        }
    
    # Build comprehensive file context
    file_content = self._build_file_context(file_chunks)
    
    # Create improvement-specific prompts
    improvement_prompts = {
        "all": """
        Analyze this code file and provide comprehensive improvement suggestions:
        
        Focus on ALL aspects:
        🎨 Readability & Style, ⚡ Performance, 🔒 Security, 🛠️ Maintainability
        
        For each suggestion, provide:
        - Specific issue description
        - Recommended solution
        - Code example (before/after)
        - Priority level (High/Medium/Low)
        - Estimated effort (Small/Medium/Large)
        - Expected impact
        """,
        
        "performance": """
        Focus specifically on performance optimizations and efficiency:
        
        Analyze for:
        - Algorithm efficiency and Big O improvements
        - Memory usage optimization
        - I/O operation efficiency
        - Database query optimization
        - Caching opportunities
        - Computational bottlenecks
        
        Provide specific performance improvements with measurable impact.
        """,
        
        "readability": """
        Focus on code readability, clarity, and maintainability:
        
        Analyze for:
        - Naming conventions and clarity
        - Code organization and structure
        - Function and class design
        - Documentation and comments
        - Code complexity reduction
        - Simplification opportunities
        
        Suggest improvements that make the code easier to understand and modify.
        """,
        
        "security": """
        Focus specifically on security vulnerabilities and improvements:
        
        Analyze for:
        - Input validation and sanitization
        - Authentication and authorization
        - Data protection and encryption
        - Error handling and information leakage
        - Dependency security issues
        - Common vulnerability patterns (OWASP Top 10)
        
        Identify specific security risks and provide mitigation strategies.
        """,
        
        "maintainability": """
        Focus on long-term code maintainability and extensibility:
        
        Analyze for:
        - Code duplication and DRY principles
        - Modularity and separation of concerns
        - Testing coverage and quality
        - Documentation completeness
        - Extensibility and flexibility
        - Technical debt reduction
        
        Suggest improvements that make the code easier to maintain and extend.
        """
    }
    
    # Create the analysis prompt
    prompt = f"""
    Analyze this code file and provide specific improvement suggestions:

    File: {file_path}
    Type of Analysis: {improvement_type}

    Code Content:
    ```
    {file_content}
    ```

    {improvement_prompts.get(improvement_type, improvement_prompts["all"])}

    Structure your response as:

    🎯 **Critical Issues** (Fix immediately)
    - Issue 1: [Description]
      - Solution: [How to fix]
      - Example: [Code before/after]
      - Impact: [Expected improvement]

    📋 **Improvement Opportunities** (Address soon)
    - Opportunity 1: [Description]
      - Recommendation: [What to do]
      - Effort: [Small/Medium/Large]
      - Benefit: [Why it matters]

    💡 **Enhancement Suggestions** (Consider for future)
    - Enhancement 1: [Description]
      - Approach: [How to implement]
      - Value: [Long-term benefit]

    📊 **Summary**
    - Overall quality score: [1-10]
    - Top priority: [Most important fix]
    - Quick wins: [Easy improvements]
    """
    
    try:
        # Execute AI analysis
        response = self._run_ai_analysis(prompt, "code_suggestions")
        
        # Structure the response with metadata
        structured_response = {
            "analysis": response.get("analysis", "Analysis not available"),
            "file_path": file_path,
            "improvement_type": improvement_type,
            "chunks_analyzed": len(file_chunks),
            "file_size_lines": sum(chunk.end_line - chunk.start_line + 1 for chunk in file_chunks),
            "timestamp": response.get("timestamp"),
            "provider": response.get("provider"),
            "cached": response.get("cached", False)
        }
        
        # Add file metadata
        file_stats = {
            "functions": len([c for c in file_chunks if c.chunk_type == CodeChunkType.FUNCTION]),
            "classes": len([c for c in file_chunks if c.chunk_type == CodeChunkType.CLASS]),
            "avg_complexity": sum(c.complexity_score for c in file_chunks) / len(file_chunks),
            "has_docstrings": sum(1 for c in file_chunks if c.docstring) / len(file_chunks)
        }
        
        structured_response["file_stats"] = file_stats
        
        logger.info(f"✅ Code suggestions generated for {file_path}")
        return structured_response
        
    except Exception as e:
        logger.error(f"❌ Code suggestions failed for {file_path}: {e}")
        return {
            "error": str(e),
            "file_path": file_path,
            "improvement_type": improvement_type,
            "timestamp": self._get_timestamp()
        }

# Usage examples:

# Comprehensive analysis
suggestions = codebase_agent.get_code_suggestions(
    file_path="src/data_analyzer.py",
    improvement_type="all"
)
print(f"🔧 Comprehensive Code Analysis:")
print(f"File: {suggestions['file_path']}")
print(f"Functions: {suggestions['file_stats']['functions']}")
print(f"Classes: {suggestions['file_stats']['classes']}")
print(f"\n{suggestions['analysis']}")

# Performance-focused analysis
perf_suggestions = codebase_agent.get_code_suggestions(
    file_path="src/slow_processor.py",
    improvement_type="performance"
)
print(f"⚡ Performance Improvements:")
print(perf_suggestions['analysis'])

# Security analysis
security_suggestions = codebase_agent.get_code_suggestions(
    file_path="src/auth_handler.py",
    improvement_type="security"
)
print(f"🔒 Security Analysis:")
print(security_suggestions['analysis'])
```

## 🔧 Language-Specific Chunking

### Python Chunking (AST-Based)

**Precise Python code analysis using Abstract Syntax Trees**

```python
def _chunk_python_file(self, file_path: Path, content: str) -> List[CodeChunk]:
    """
    Chunk Python file using AST parsing for maximum precision
    
    Extracts:
    - Function definitions with arguments and docstrings
    - Class definitions with inheritance and methods
    - Module-level constants and variables
    - Import statements and dependencies
    - Decorator usage and metadata
    """
    chunks = []
    
    try:
        # Parse Python code into AST
        tree = ast.parse(content)
        
        # Walk through all AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function information
                func_content = ast.get_source_segment(content, node)
                if func_content:
                    
                    # Calculate complexity (simplified cyclomatic complexity)
                    complexity = self._calculate_python_complexity(node)
                    
                    # Extract function metadata
                    metadata = {
                        "args": [arg.arg for arg in node.args.args],
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "line_count": node.end_lineno - node.lineno + 1
                    }
                    
                    chunks.append(CodeChunk(
                        id=f"{file_path}:{node.name}",
                        file_path=str(file_path),
                        chunk_type=CodeChunkType.FUNCTION,
                        name=node.name,
                        content=func_content,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        complexity_score=complexity,
                        metadata=metadata
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Extract class information
                class_content = ast.get_source_segment(content, node)
                if class_content:
                    
                    # Extract class metadata
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    metadata = {
                        "bases": [ast.unparse(base) for base in node.bases],
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "methods": methods,
                        "method_count": len(methods),
                        "line_count": node.end_lineno - node.lineno + 1
                    }
                    
                    chunks.append(CodeChunk(
                        id=f"{file_path}:{node.name}",
                        file_path=str(file_path),
                        chunk_type=CodeChunkType.CLASS,
                        name=node.name,
                        content=class_content,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        complexity_score=len(methods) * 0.1,  # Simple class complexity
                        metadata=metadata
                    ))
    
    except SyntaxError as e:
        logger.warning(f"Python syntax error in {file_path}: {e}")
        # Fallback to generic text chunking
        chunks.extend(self._chunk_generic_file(file_path, content))
    
    return chunks

def _calculate_python_complexity(self, node: ast.AST) -> float:
    """Calculate simplified cyclomatic complexity for Python functions"""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        # Count decision points
        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    
    # Normalize to 0-1 scale
    return min(1.0, complexity / 10.0)
```

### JavaScript/TypeScript Chunking

**Pattern-based chunking for modern JavaScript**

```python
def _chunk_javascript_file(self, file_path: Path, content: str) -> List[CodeChunk]:
    """
    Chunk JavaScript/TypeScript files using pattern matching
    
    Detects:
    - Function declarations and expressions
    - Arrow functions and async functions
    - Class definitions and methods
    - ES6 imports and exports
    - JSDoc comments and TypeScript types
    """
    chunks = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect function declarations
        if self._is_js_function_start(line):
            chunk_lines, chunk_end = self._extract_js_block(lines, i)
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                function_name = self._extract_js_function_name(chunk_content)
                
                # Extract metadata
                metadata = {
                    "is_async": "async" in chunk_lines[0],
                    "is_arrow_function": "=>" in chunk_lines[0],
                    "has_jsdoc": any("/**" in l for l in chunk_lines[:5]),
                    "line_count": len(chunk_lines)
                }
                
                chunks.append(CodeChunk(
                    id=f"{file_path}:{function_name}:{i}",
                    file_path=str(file_path),
                    chunk_type=CodeChunkType.FUNCTION,
                    name=function_name,
                    content=chunk_content,
                    start_line=i + 1,
                    end_line=chunk_end + 1,
                    metadata=metadata
                ))
            
            i = chunk_end + 1
        else:
            i += 1
    
    return chunks

def _is_js_function_start(self, line: str) -> bool:
    """Detect various JavaScript function patterns"""
    patterns = [
        r'\bfunction\s+\w+',           # function name()
        r'\w+\s*=\s*function',         # name = function
        r'\w+\s*:\s*function',         # name: function
        r'class\s+\w+',                # class Name
        r'\w+\s*=\s*.*=>',             # name = () =>
        r'async\s+function',           # async function
        r'export\s+function',          # export function
    ]
    
    return any(re.search(pattern, line) for pattern in patterns)
```

## 🚀 Usage Examples

### Basic Codebase Analysis Workflow

```python
from AI_Functionality.core.codebase_agent import CodebaseAgent
from AI_Functionality.core.ai_analyst import DataAnalyst

# Initialize components
analyst = DataAnalyst(primary_provider="openai")
codebase_agent = CodebaseAgent(
    codebase_path="./my_project",
    ai_analyst=analyst,
    cache_dir="./codebase_cache"
)

# Step 1: Chunk the codebase
print("📁 Chunking codebase...")
chunk_count = codebase_agent.chunk_codebase(
    progress_callback=lambda msg, pct: print(f"{msg} ({pct:.1%})")
)
print(f"✅ Created {chunk_count} chunks")

# Step 2: Get codebase overview
print("\n🔍 Analyzing codebase overview...")
overview = codebase_agent.analyze_codebase("overview")
print(f"📊 Overview Analysis:")
print(overview['analysis'])

# Step 3: Ask specific questions
print("\n❓ Interactive Q&A:")
questions = [
    "How does user authentication work?",
    "Where are the API endpoints defined?",
    "What testing framework is used?",
    "How is configuration managed?"
]

for question in questions:
    response = codebase_agent.answer_codebase_question(question)
    print(f"Q: {question}")
    print(f"A: {response['answer'][:200]}...")
    print(f"📁 Files: {', '.join(response['relevant_files'][:3])}")
    print()

# Step 4: Get improvement suggestions
print("\n🔧 Code Improvement Suggestions:")
main_files = ["src/main.py", "src/api.py", "src/models.py"]

for file_path in main_files:
    suggestions = codebase_agent.get_code_suggestions(file_path, "all")
    if 'error' not in suggestions:
        print(f"📄 {file_path}:")
        print(f"   Functions: {suggestions['file_stats']['functions']}")
        print(f"   Complexity: {suggestions['file_stats']['avg_complexity']:.2f}")
        print(f"   {suggestions['analysis'][:150]}...")
        print()
```

### Comprehensive Architecture Analysis

```python
# Deep architecture analysis
print("🏗️ Architecture Deep Dive:")
architecture = codebase_agent.analyze_codebase("architecture")

print(f"Analysis Results:")
print(f"Files analyzed: {architecture['file_count']}")
print(f"Chunks processed: {architecture['chunk_count']}")
print(f"\n{architecture['analysis']}")

# Security assessment
print("\n🔒 Security Analysis:")
security = codebase_agent.analyze_codebase("security")
print(security['analysis'])

# Quality evaluation
print("\n📊 Quality Assessment:")
quality = codebase_agent.analyze_codebase("quality")
print(quality['analysis'])
```

### File-Specific Analysis

```python
# Analyze specific files for improvements
critical_files = [
    "src/auth.py",
    "src/database.py", 
    "src/api_handlers.py"
]

print("🎯 File-Specific Analysis:")
for file_path in critical_files:
    print(f"\n📄 Analyzing {file_path}:")
    
    # General suggestions
    general = codebase_agent.get_code_suggestions(file_path, "all")
    
    # Performance focus
    performance = codebase_agent.get_code_suggestions(file_path, "performance")
    
    # Security focus  
    security = codebase_agent.get_code_suggestions(file_path, "security")
    
    if 'error' not in general:
        print(f"📊 General: {general['analysis'][:100]}...")
        print(f"⚡ Performance: {performance['analysis'][:100]}...")
        print(f"🔒 Security: {security['analysis'][:100]}...")
```

### Monitoring and Statistics

```python
# Get codebase statistics
stats = codebase_agent.get_codebase_stats()
print("📊 Codebase Statistics:")
print(f"Total files: {stats['total_files']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Chunk types: {stats['chunk_types']}")
print(f"File extensions: {stats['file_extensions']}")
print(f"Largest files: {stats['largest_files'][:5]}")

# Monitor cache performance
cache_stats = codebase_agent.cache_manager.get_cache_stats()
print(f"\n💾 Cache Performance:")
print(f"Semantic caching: {'✅' if cache_stats['enabled_features']['semantic_caching'] else '❌'}")
print(f"Total cache entries: {sum(v for k, v in cache_stats['cache_sizes'].items() if k.endswith('_entries'))}")
```

This comprehensive documentation covers all aspects of the CodebaseAgent, from intelligent code chunking to AI-powered analysis and interactive Q&A capabilities.