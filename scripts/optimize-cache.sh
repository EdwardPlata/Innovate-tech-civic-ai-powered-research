#!/bin/bash

# Cache Optimizer Script
# Cleans and optimizes cache directories for Scout Data Discovery

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info() {
    echo -e "${BLUE}[CACHE-OPT]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to get directory size
get_dir_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to count files in directory
count_files() {
    local dir=$1
    if [ -d "$dir" ]; then
        find "$dir" -type f 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

print_info "=== Cache Optimizer ==="
echo

# Analyze cache directories
print_info "Analyzing cache directories..."

BACKEND_CACHE="$PROJECT_ROOT/backend/cache"
ROOT_CACHE="$PROJECT_ROOT/cache"
PYCACHE_DIRS=$(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null)

# Backend cache
if [ -d "$BACKEND_CACHE" ]; then
    BACKEND_SIZE=$(get_dir_size "$BACKEND_CACHE")
    BACKEND_FILES=$(count_files "$BACKEND_CACHE")
    print_info "Backend cache: $BACKEND_SIZE ($BACKEND_FILES files)"
fi

# Root cache
if [ -d "$ROOT_CACHE" ]; then
    ROOT_SIZE=$(get_dir_size "$ROOT_CACHE")
    ROOT_FILES=$(count_files "$ROOT_CACHE")
    print_info "Root cache: $ROOT_SIZE ($ROOT_FILES files)"
fi

# Python cache
PYCACHE_COUNT=$(echo "$PYCACHE_DIRS" | grep -c . 2>/dev/null || echo "0")
if [ "$PYCACHE_COUNT" -gt 0 ]; then
    print_info "Python __pycache__ directories: $PYCACHE_COUNT"
fi

echo

# Clean old cache files (older than 7 days)
if [ "$1" = "--clean-old" ] || [ "$1" = "-o" ]; then
    print_info "Cleaning cache files older than 7 days..."
    
    if [ -d "$BACKEND_CACHE" ]; then
        OLD_FILES=$(find "$BACKEND_CACHE" -type f -mtime +7 2>/dev/null | wc -l)
        if [ "$OLD_FILES" -gt 0 ]; then
            find "$BACKEND_CACHE" -type f -mtime +7 -delete 2>/dev/null || true
            print_success "Removed $OLD_FILES old files from backend cache"
        else
            print_info "No old files found in backend cache"
        fi
    fi
    
    if [ -d "$ROOT_CACHE" ]; then
        OLD_FILES=$(find "$ROOT_CACHE" -type f -mtime +7 2>/dev/null | wc -l)
        if [ "$OLD_FILES" -gt 0 ]; then
            find "$ROOT_CACHE" -type f -mtime +7 -delete 2>/dev/null || true
            print_success "Removed $OLD_FILES old files from root cache"
        else
            print_info "No old files found in root cache"
        fi
    fi
fi

# Clean all cache
if [ "$1" = "--clean-all" ] || [ "$1" = "-a" ]; then
    print_warning "Cleaning all cache directories..."
    
    if [ -d "$BACKEND_CACHE" ]; then
        rm -rf "$BACKEND_CACHE"/*
        print_success "Backend cache cleaned"
    fi
    
    if [ -d "$ROOT_CACHE" ]; then
        rm -rf "$ROOT_CACHE"/*
        print_success "Root cache cleaned"
    fi
fi

# Clean Python cache
if [ "$1" = "--clean-python" ] || [ "$1" = "-p" ] || [ "$1" = "--clean-all" ]; then
    print_info "Cleaning Python __pycache__ directories..."
    
    if [ "$PYCACHE_COUNT" -gt 0 ]; then
        # Using while loop for safer handling of paths with special characters
        find "$PROJECT_ROOT" -type d -name "__pycache__" -print0 2>/dev/null | while IFS= read -r -d '' dir; do
            rm -rf "$dir" 2>/dev/null || true
        done
        find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
        find "$PROJECT_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true
        print_success "Python cache cleaned"
    else
        print_info "No Python cache directories found"
    fi
fi

# Optimize cache structure
if [ "$1" = "--optimize" ]; then
    print_info "Optimizing cache structure..."
    
    # Ensure cache directories exist
    mkdir -p "$BACKEND_CACHE/datasets"
    mkdir -p "$BACKEND_CACHE/relationships"
    mkdir -p "$BACKEND_CACHE/quality"
    
    print_success "Cache structure optimized"
fi

# Show cache statistics
if [ "$1" = "--stats" ] || [ "$1" = "-s" ]; then
    echo
    print_info "=== Cache Statistics ==="
    
    if [ -d "$BACKEND_CACHE/datasets" ]; then
        DATASETS=$(count_files "$BACKEND_CACHE/datasets")
        print_info "Cached datasets: $DATASETS"
    fi
    
    if [ -d "$BACKEND_CACHE/relationships" ]; then
        RELATIONSHIPS=$(count_files "$BACKEND_CACHE/relationships")
        print_info "Cached relationships: $RELATIONSHIPS"
    fi
    
    if [ -d "$BACKEND_CACHE/quality" ]; then
        QUALITY=$(count_files "$BACKEND_CACHE/quality")
        print_info "Cached quality assessments: $QUALITY"
    fi
fi

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --clean-old, -o      Clean cache files older than 7 days"
    echo "  --clean-all, -a      Clean all cache directories"
    echo "  --clean-python, -p   Clean Python __pycache__ directories"
    echo "  --optimize           Optimize cache directory structure"
    echo "  --stats, -s          Show cache statistics"
    echo "  --help, -h           Show this help message"
    echo
    exit 0
fi

echo
print_success "=== Cache optimization complete ==="
