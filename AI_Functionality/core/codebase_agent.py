"""
Codebase Agent - AI-powered code analysis and Q&A

Provides intelligent codebase understanding through chunking, analysis,
and interactive Q&A capabilities.
"""

import os
import ast
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from .base_provider import BaseAIProvider, AIRequest, AIResponse
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class CodeChunkType(Enum):
    """Types of code chunks for analysis"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    IMPORT = "import"
    COMMENT = "comment"
    CONFIG = "config"


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    id: str
    file_path: str
    chunk_type: CodeChunkType
    name: str
    content: str
    start_line: int
    end_line: int
    dependencies: List[str] = None
    docstring: Optional[str] = None
    complexity_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class CodebaseAgent:
    """AI-powered codebase analysis and Q&A agent"""

    def __init__(
        self,
        codebase_path: str,
        ai_analyst,
        cache_dir: str = "./codebase_cache",
        supported_extensions: List[str] = None
    ):
        """
        Initialize Codebase Agent

        Args:
            codebase_path: Path to the codebase to analyze
            ai_analyst: DataAnalyst instance for AI operations
            cache_dir: Directory for caching chunked code
            supported_extensions: File extensions to analyze
        """
        self.codebase_path = Path(codebase_path)
        self.ai_analyst = ai_analyst
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Initialize cache manager for codebase analysis
        self.cache_manager = CacheManager(
            cache_dir=str(self.cache_dir),
            enable_semantic=True
        )

        self.supported_extensions = supported_extensions or [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.php', '.rb', '.rs', '.sql', '.md', '.yml', '.yaml', '.json'
        ]

        # Codebase analysis state
        self.chunks: List[CodeChunk] = []
        self.file_index: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.codebase_summary: Optional[Dict[str, Any]] = None

        logger.info(f"CodebaseAgent initialized for: {self.codebase_path}")

    def chunk_codebase(self, progress_callback: Optional[callable] = None) -> int:
        """
        Chunk the codebase into analyzable pieces

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Number of chunks created
        """
        logger.info("Starting codebase chunking...")

        if progress_callback:
            progress_callback("Scanning codebase files...", 0.1)

        # Find all relevant files
        files_to_analyze = self._find_code_files()

        if progress_callback:
            progress_callback(f"Found {len(files_to_analyze)} files to analyze", 0.2)

        # Check for cached chunks
        cache_key = self._get_codebase_cache_key(files_to_analyze)
        cached_chunks = self._load_cached_chunks(cache_key)

        if cached_chunks:
            self.chunks = cached_chunks
            logger.info(f"Loaded {len(self.chunks)} cached chunks")
            if progress_callback:
                progress_callback("Loaded cached chunks", 1.0)
            return len(self.chunks)

        # Process files and create chunks
        self.chunks = []
        total_files = len(files_to_analyze)

        for i, file_path in enumerate(files_to_analyze):
            if progress_callback:
                progress = 0.2 + (0.7 * (i / total_files))
                progress_callback(f"Processing {file_path.name}...", progress)

            try:
                file_chunks = self._chunk_file(file_path)
                self.chunks.extend(file_chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")

        # Cache the chunks
        self._cache_chunks(cache_key, self.chunks)

        # Build indexes
        if progress_callback:
            progress_callback("Building indexes...", 0.9)

        self._build_indexes()

        logger.info(f"Codebase chunking complete: {len(self.chunks)} chunks created")

        if progress_callback:
            progress_callback("Chunking complete!", 1.0)

        return len(self.chunks)

    def analyze_codebase(self, analysis_focus: str = "overview") -> Dict[str, Any]:
        """
        Perform high-level codebase analysis using AI

        Args:
            analysis_focus: Focus of analysis (overview, architecture, quality, security)

        Returns:
            Analysis results
        """
        if not self.chunks:
            raise ValueError("Codebase not chunked. Call chunk_codebase() first.")

        logger.info(f"Starting codebase analysis: {analysis_focus}")

        # Check cache
        cache_key = f"codebase_analysis_{analysis_focus}_{len(self.chunks)}"
        cached_analysis = self.cache_manager.get_cached_context(cache_key)

        if cached_analysis:
            logger.info("Using cached codebase analysis")
            return cached_analysis

        # Prepare analysis context
        context = self._build_codebase_context()

        # Create AI analysis prompt
        analysis_prompts = {
            "overview": f"""
            Analyze this codebase and provide a comprehensive overview:

            {context}

            Please provide:
            1. **Architecture Overview**: High-level structure and patterns
            2. **Key Components**: Main modules, classes, and their purposes
            3. **Technology Stack**: Languages, frameworks, and tools used
            4. **Code Organization**: Directory structure and naming conventions
            5. **Dependencies**: External libraries and internal module relationships
            6. **Complexity Assessment**: Overall complexity and maintainability
            7. **Recommendations**: Suggestions for improvement or areas of focus

            Focus on actionable insights for developers working with this codebase.
            """,

            "architecture": f"""
            Analyze the architecture and design patterns of this codebase:

            {context}

            Focus on:
            1. **Design Patterns**: Identify architectural patterns used
            2. **Module Relationships**: How components interact
            3. **Data Flow**: How data moves through the system
            4. **Separation of Concerns**: How responsibilities are divided
            5. **Scalability**: Architecture's ability to scale
            6. **Maintainability**: Ease of modification and extension
            7. **Technical Debt**: Potential issues and refactoring opportunities

            Provide specific recommendations for architectural improvements.
            """,

            "quality": f"""
            Assess the code quality and maintainability of this codebase:

            {context}

            Evaluate:
            1. **Code Style**: Consistency and adherence to conventions
            2. **Documentation**: Quality of comments and docstrings
            3. **Testing**: Test coverage and quality
            4. **Error Handling**: Robustness and error management
            5. **Performance**: Potential bottlenecks and optimizations
            6. **Security**: Security considerations and vulnerabilities
            7. **Maintainability Score**: Overall maintainability rating (1-10)

            Highlight specific files or components that need attention.
            """,

            "security": f"""
            Perform a security analysis of this codebase:

            {context}

            Examine:
            1. **Input Validation**: How user input is handled
            2. **Authentication/Authorization**: Security mechanisms
            3. **Data Protection**: Sensitive data handling
            4. **Dependency Security**: Third-party library risks
            5. **Configuration Security**: Secure configuration practices
            6. **Common Vulnerabilities**: OWASP Top 10 considerations
            7. **Security Recommendations**: Specific improvements needed

            Focus on identifying potential security vulnerabilities and mitigation strategies.
            """
        }

        prompt = analysis_prompts.get(analysis_focus, analysis_prompts["overview"])

        try:
            # Run AI analysis
            analysis_result = self._run_ai_analysis(prompt, analysis_focus)

            # Cache the result
            self.cache_manager.cache_context_analysis(cache_key, analysis_result)

            return analysis_result

        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_focus,
                "timestamp": self._get_timestamp()
            }

    def answer_codebase_question(self, question: str, context_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Answer questions about the codebase using AI

        Args:
            question: User's question about the codebase
            context_files: Optional list of specific files to focus on

        Returns:
            Answer and relevant code references
        """
        if not self.chunks:
            raise ValueError("Codebase not chunked. Call chunk_codebase() first.")

        logger.info(f"Answering codebase question: {question[:100]}...")

        # Find relevant chunks for the question
        relevant_chunks = self._find_relevant_chunks(question, context_files)

        # Build context from relevant chunks
        context = self._build_question_context(relevant_chunks, question)

        # Create AI prompt
        prompt = f"""
        Based on the following codebase information, please answer this question: "{question}"

        Codebase Context:
        {context}

        Please provide:
        1. **Direct Answer**: Clear answer to the question
        2. **Code References**: Specific files and functions involved
        3. **Explanation**: How the code works to address the question
        4. **Related Components**: Other parts of the codebase that might be relevant
        5. **Examples**: Code snippets if helpful
        6. **Recommendations**: Any suggestions for improvement

        Be specific and reference actual code elements when possible.
        """

        try:
            # Run AI analysis
            response = self._run_ai_analysis(prompt, "question_answer")

            # Add metadata about the search
            response.update({
                "question": question,
                "relevant_files": [chunk.file_path for chunk in relevant_chunks],
                "chunks_analyzed": len(relevant_chunks),
                "search_scope": context_files or "entire_codebase"
            })

            return response

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                "error": str(e),
                "question": question,
                "timestamp": self._get_timestamp()
            }

    def get_code_suggestions(self, file_path: str, improvement_type: str = "all") -> Dict[str, Any]:
        """
        Get AI-powered suggestions for improving specific files

        Args:
            file_path: Path to the file to analyze
            improvement_type: Type of improvements (all, performance, readability, security)

        Returns:
            Improvement suggestions
        """
        # Find chunks for the specific file
        file_chunks = [chunk for chunk in self.chunks if chunk.file_path == file_path]

        if not file_chunks:
            return {"error": f"File not found in codebase: {file_path}"}

        # Build file context
        file_content = "\n\n".join([
            f"# {chunk.name} ({chunk.chunk_type.value})\n{chunk.content}"
            for chunk in file_chunks
        ])

        improvement_prompts = {
            "all": "Analyze this code and suggest improvements for readability, performance, security, and maintainability.",
            "performance": "Focus on performance optimizations and efficiency improvements.",
            "readability": "Suggest improvements for code readability, naming, and documentation.",
            "security": "Identify potential security vulnerabilities and suggest fixes.",
            "maintainability": "Suggest changes to improve code maintainability and extensibility."
        }

        prompt = f"""
        Analyze this code file and provide specific improvement suggestions:

        File: {file_path}

        Code:
        ```
        {file_content}
        ```

        {improvement_prompts.get(improvement_type, improvement_prompts["all"])}

        Provide:
        1. **Specific Issues**: Line-by-line issues identified
        2. **Improvement Suggestions**: Concrete recommendations
        3. **Code Examples**: Before/after code snippets
        4. **Priority**: High/Medium/Low priority for each suggestion
        5. **Impact**: Expected impact of each improvement
        """

        try:
            response = self._run_ai_analysis(prompt, "code_suggestions")
            response.update({
                "file_path": file_path,
                "improvement_type": improvement_type,
                "chunks_analyzed": len(file_chunks)
            })
            return response

        except Exception as e:
            logger.error(f"Code suggestions failed: {e}")
            return {
                "error": str(e),
                "file_path": file_path,
                "timestamp": self._get_timestamp()
            }

    def _find_code_files(self) -> List[Path]:
        """Find all relevant code files in the codebase"""
        code_files = []

        # Skip common directories to ignore
        ignore_dirs = {
            '.git', '.venv', 'venv', '__pycache__', 'node_modules',
            '.pytest_cache', 'build', 'dist', '.idea', '.vscode',
            'cache', 'logs', 'tmp', 'temp'
        }

        for root, dirs, files in os.walk(self.codebase_path):
            # Remove ignored directories from traversal
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    # Skip very large files (>1MB)
                    try:
                        if file_path.stat().st_size < 1024 * 1024:
                            code_files.append(file_path)
                    except (OSError, IOError):
                        continue

        return sorted(code_files)

    def _chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """Chunk a single file into analyzable pieces"""
        chunks = []

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return chunks

        # Handle different file types
        if file_path.suffix == '.py':
            chunks.extend(self._chunk_python_file(file_path, content))
        elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            chunks.extend(self._chunk_javascript_file(file_path, content))
        elif file_path.suffix in ['.md', '.txt', '.rst']:
            chunks.extend(self._chunk_text_file(file_path, content))
        elif file_path.suffix in ['.json', '.yml', '.yaml']:
            chunks.extend(self._chunk_config_file(file_path, content))
        else:
            # Generic text chunking
            chunks.extend(self._chunk_generic_file(file_path, content))

        return chunks

    def _chunk_python_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk Python file using AST parsing"""
        chunks = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Function chunk
                    func_content = ast.get_source_segment(content, node)
                    if func_content:
                        chunks.append(CodeChunk(
                            id=f"{file_path}:{node.name}",
                            file_path=str(file_path),
                            chunk_type=CodeChunkType.FUNCTION,
                            name=node.name,
                            content=func_content,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            docstring=ast.get_docstring(node),
                            metadata={"args": [arg.arg for arg in node.args.args]}
                        ))

                elif isinstance(node, ast.ClassDef):
                    # Class chunk
                    class_content = ast.get_source_segment(content, node)
                    if class_content:
                        chunks.append(CodeChunk(
                            id=f"{file_path}:{node.name}",
                            file_path=str(file_path),
                            chunk_type=CodeChunkType.CLASS,
                            name=node.name,
                            content=class_content,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            docstring=ast.get_docstring(node),
                            metadata={"bases": [ast.unparse(base) for base in node.bases]}
                        ))

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fall back to generic chunking
            chunks.extend(self._chunk_generic_file(file_path, content))

        return chunks

    def _chunk_javascript_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript files"""
        # Simple regex-based chunking for JS/TS
        # In a production system, you'd want to use a proper JS parser
        chunks = []

        lines = content.split('\n')
        current_chunk = []
        in_function = False
        brace_count = 0
        chunk_start = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Simple detection of function starts
            if 'function ' in line or '=>' in line or 'class ' in line:
                if not in_function:
                    in_function = True
                    chunk_start = i

            # Count braces to detect function/class end
            brace_count += line.count('{') - line.count('}')

            if in_function and brace_count <= 0 and '{' in ''.join(current_chunk):
                # End of function/class
                chunk_content = '\n'.join(current_chunk)
                chunk_name = self._extract_js_function_name(chunk_content)

                chunks.append(CodeChunk(
                    id=f"{file_path}:{chunk_name}:{chunk_start}",
                    file_path=str(file_path),
                    chunk_type=CodeChunkType.FUNCTION,
                    name=chunk_name,
                    content=chunk_content,
                    start_line=chunk_start + 1,
                    end_line=i + 1
                ))

                current_chunk = []
                in_function = False
                brace_count = 0

        return chunks

    def _chunk_text_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk text/markdown files"""
        chunks = []

        # Split by headers for markdown
        if file_path.suffix == '.md':
            sections = content.split('#')
            for i, section in enumerate(sections):
                if section.strip():
                    header_line = section.split('\n')[0].strip()
                    chunks.append(CodeChunk(
                        id=f"{file_path}:section_{i}",
                        file_path=str(file_path),
                        chunk_type=CodeChunkType.MODULE,
                        name=header_line or f"Section {i}",
                        content=section.strip(),
                        start_line=0,
                        end_line=len(section.split('\n'))
                    ))
        else:
            # Generic text chunking by paragraphs
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunks.append(CodeChunk(
                        id=f"{file_path}:paragraph_{i}",
                        file_path=str(file_path),
                        chunk_type=CodeChunkType.COMMENT,
                        name=f"Paragraph {i+1}",
                        content=paragraph.strip(),
                        start_line=0,
                        end_line=len(paragraph.split('\n'))
                    ))

        return chunks

    def _chunk_config_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk configuration files"""
        chunks = []

        # Single chunk for config files
        chunks.append(CodeChunk(
            id=str(file_path),
            file_path=str(file_path),
            chunk_type=CodeChunkType.CONFIG,
            name=file_path.name,
            content=content,
            start_line=1,
            end_line=len(content.split('\n'))
        ))

        return chunks

    def _chunk_generic_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Generic file chunking by size"""
        chunks = []
        lines = content.split('\n')
        chunk_size = 100  # Lines per chunk

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)

            chunks.append(CodeChunk(
                id=f"{file_path}:chunk_{i // chunk_size}",
                file_path=str(file_path),
                chunk_type=CodeChunkType.MODULE,
                name=f"{file_path.name} (chunk {i // chunk_size + 1})",
                content=chunk_content,
                start_line=i + 1,
                end_line=i + len(chunk_lines)
            ))

        return chunks

    def _extract_js_function_name(self, content: str) -> str:
        """Extract function name from JavaScript code"""
        import re

        # Try different patterns
        patterns = [
            r'function\s+(\w+)',  # function name()
            r'(\w+)\s*=\s*function',  # name = function
            r'(\w+)\s*:\s*function',  # name: function
            r'class\s+(\w+)',  # class Name
            r'(\w+)\s*=\s*.*=>',  # name = () =>
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        return "anonymous"

    def _build_codebase_context(self) -> str:
        """Build comprehensive context about the codebase"""
        # Codebase statistics
        stats = {
            "total_files": len(set(chunk.file_path for chunk in self.chunks)),
            "total_chunks": len(self.chunks),
            "functions": len([c for c in self.chunks if c.chunk_type == CodeChunkType.FUNCTION]),
            "classes": len([c for c in self.chunks if c.chunk_type == CodeChunkType.CLASS]),
            "modules": len([c for c in self.chunks if c.chunk_type == CodeChunkType.MODULE]),
        }

        # File extensions distribution
        extensions = {}
        for chunk in self.chunks:
            ext = Path(chunk.file_path).suffix
            extensions[ext] = extensions.get(ext, 0) + 1

        # Sample of important chunks
        important_chunks = sorted(
            [c for c in self.chunks if c.chunk_type in [CodeChunkType.CLASS, CodeChunkType.FUNCTION]],
            key=lambda x: len(x.content),
            reverse=True
        )[:10]

        context = f"""
        Codebase Analysis Summary:
        - Total Files: {stats['total_files']}
        - Functions: {stats['functions']}
        - Classes: {stats['classes']}
        - File Types: {', '.join(f"{ext}: {count}" for ext, count in extensions.items())}

        Key Components:
        """

        for chunk in important_chunks:
            context += f"\n{chunk.name} ({chunk.chunk_type.value}): {chunk.file_path}"

        context += "\n\nSample Code Structure:\n"

        for chunk in important_chunks[:5]:
            preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            context += f"\n--- {chunk.name} ---\n{preview}\n"

        return context

    def _find_relevant_chunks(self, question: str, context_files: Optional[List[str]] = None) -> List[CodeChunk]:
        """Find code chunks relevant to the question"""
        relevant_chunks = []

        # Filter by context files if specified
        if context_files:
            relevant_chunks = [
                chunk for chunk in self.chunks
                if any(cf in chunk.file_path for cf in context_files)
            ]
        else:
            relevant_chunks = self.chunks

        # Simple relevance scoring based on keyword matching
        question_words = set(question.lower().split())
        scored_chunks = []

        for chunk in relevant_chunks:
            score = 0

            # Check name match
            if any(word in chunk.name.lower() for word in question_words):
                score += 3

            # Check content match
            content_words = set(chunk.content.lower().split())
            matches = question_words.intersection(content_words)
            score += len(matches)

            # Check docstring match
            if chunk.docstring:
                docstring_words = set(chunk.docstring.lower().split())
                doc_matches = question_words.intersection(docstring_words)
                score += len(doc_matches) * 2

            if score > 0:
                scored_chunks.append((chunk, score))

        # Sort by relevance and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:20]]  # Top 20 most relevant

    def _build_question_context(self, chunks: List[CodeChunk], question: str) -> str:
        """Build context from relevant chunks for question answering"""
        context = f"Question: {question}\n\nRelevant Code:\n"

        for chunk in chunks:
            context += f"\n--- {chunk.name} ({chunk.file_path}) ---\n"
            context += chunk.content
            if chunk.docstring:
                context += f"\nDocstring: {chunk.docstring}"
            context += "\n"

        return context

    def _run_ai_analysis(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Run AI analysis using the DataAnalyst"""
        import asyncio

        try:
            # Create a mock dataset info for AI analysis
            dataset_info = {
                "id": f"codebase_{analysis_type}",
                "name": f"Codebase {analysis_type.title()} Analysis",
                "description": f"AI-powered {analysis_type} analysis of the codebase",
                "category": "Code Analysis"
            }

            # Run async analysis
            response = asyncio.run(self.ai_analyst.answer_question(
                question=prompt,
                dataset_info=dataset_info,
                use_cache=True
            ))

            return {
                "analysis": response.content if hasattr(response, 'content') else str(response),
                "analysis_type": analysis_type,
                "timestamp": self._get_timestamp(),
                "provider": response.provider if hasattr(response, 'provider') else "unknown",
                "cached": response.cached if hasattr(response, 'cached') else False
            }

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            raise

    def _get_codebase_cache_key(self, files: List[Path]) -> str:
        """Generate cache key for codebase chunks"""
        # Create hash from file paths and modification times
        file_data = []
        for file_path in files:
            try:
                mtime = file_path.stat().st_mtime
                file_data.append(f"{file_path}:{mtime}")
            except (OSError, IOError):
                file_data.append(str(file_path))

        content = "|".join(sorted(file_data))
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cached_chunks(self, cache_key: str) -> Optional[List[CodeChunk]]:
        """Load cached chunks if available"""
        cache_file = self.cache_dir / f"chunks_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    chunk_data = json.load(f)

                chunks = []
                for data in chunk_data:
                    chunk = CodeChunk(
                        id=data['id'],
                        file_path=data['file_path'],
                        chunk_type=CodeChunkType(data['chunk_type']),
                        name=data['name'],
                        content=data['content'],
                        start_line=data['start_line'],
                        end_line=data['end_line'],
                        dependencies=data.get('dependencies', []),
                        docstring=data.get('docstring'),
                        complexity_score=data.get('complexity_score', 0.0),
                        metadata=data.get('metadata', {})
                    )
                    chunks.append(chunk)

                logger.info(f"Loaded {len(chunks)} chunks from cache")
                return chunks

            except Exception as e:
                logger.warning(f"Failed to load cached chunks: {e}")

        return None

    def _cache_chunks(self, cache_key: str, chunks: List[CodeChunk]):
        """Cache chunks for future use"""
        cache_file = self.cache_dir / f"chunks_{cache_key}.json"

        try:
            chunk_data = []
            for chunk in chunks:
                data = {
                    'id': chunk.id,
                    'file_path': chunk.file_path,
                    'chunk_type': chunk.chunk_type.value,
                    'name': chunk.name,
                    'content': chunk.content,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'dependencies': chunk.dependencies,
                    'docstring': chunk.docstring,
                    'complexity_score': chunk.complexity_score,
                    'metadata': chunk.metadata
                }
                chunk_data.append(data)

            with open(cache_file, 'w') as f:
                json.dump(chunk_data, f, indent=2)

            logger.info(f"Cached {len(chunks)} chunks")

        except Exception as e:
            logger.warning(f"Failed to cache chunks: {e}")

    def _build_indexes(self):
        """Build search indexes for chunks"""
        self.file_index = {}

        for chunk in self.chunks:
            if chunk.file_path not in self.file_index:
                self.file_index[chunk.file_path] = []
            self.file_index[chunk.file_path].append(chunk.id)

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_codebase_stats(self) -> Dict[str, Any]:
        """Get comprehensive codebase statistics"""
        if not self.chunks:
            return {"error": "Codebase not analyzed"}

        stats = {
            "total_files": len(self.file_index),
            "total_chunks": len(self.chunks),
            "chunk_types": {},
            "file_extensions": {},
            "largest_files": [],
            "complexity_distribution": []
        }

        # Count chunk types
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type.value
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

        # Count file extensions
        for file_path in self.file_index:
            ext = Path(file_path).suffix
            stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1

        # Find largest files (by chunk content)
        file_sizes = {}
        for chunk in self.chunks:
            file_sizes[chunk.file_path] = file_sizes.get(chunk.file_path, 0) + len(chunk.content)

        stats["largest_files"] = sorted(
            [(path, size) for path, size in file_sizes.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return stats