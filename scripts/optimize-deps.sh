#!/bin/bash

# Dependency Optimizer Script
# Checks and optimizes Python dependencies for Scout Data Discovery

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
    echo -e "${BLUE}[DEP-OPT]${NC} $1"
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

print_info "=== Dependency Optimizer ==="
echo

# Check if UV is installed
print_info "Checking UV package manager..."
if ! command -v uv &> /dev/null; then
    print_warning "UV is not installed. Installing UV..."
    pip install uv
    print_success "UV installed"
else
    UV_VERSION=$(uv --version | awk '{print $2}')
    print_info "UV version: $UV_VERSION"
fi

# Check if pip is up to date (for backwards compatibility)
print_info "Checking pip version..."
CURRENT_PIP_VERSION=$(pip --version | awk '{print $2}')
print_info "Current pip version: $CURRENT_PIP_VERSION"

# Upgrade pip if requested
if [ "$1" = "--upgrade-pip" ]; then
    print_info "Upgrading pip..."
    pip install --upgrade pip -q
    print_success "Pip upgraded"
    
    print_info "Upgrading UV..."
    pip install --upgrade uv -q
    print_success "UV upgraded"
fi

# Function to check and install requirements
check_and_install() {
    local component_dir=$1
    local name=$2
    
    local req_file="$component_dir/requirements.txt"
    
    if [ ! -f "$req_file" ]; then
        print_warning "Requirements file not found: $req_file"
        return 1
    fi
    
    print_info "Checking $name dependencies..."
    
    # Component-specific checks
    local needs_install=false
    case "$name" in
        "Backend")
            if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
                needs_install=true
            fi
            ;;
        "Frontend")
            if ! python3 -c "import streamlit" 2>/dev/null; then
                needs_install=true
            fi
            ;;
        "Scout Core")
            if ! python3 -c "import pandas, networkx" 2>/dev/null; then
                needs_install=true
            fi
            ;;
        "AI Functionality")
            if ! python3 -c "import openai" 2>/dev/null; then
                needs_install=true
            fi
            ;;
        *)
            # For unknown components, count packages
            local total_packages=$(grep -v "^#" "$req_file" | grep -v "^$" | wc -l)
            print_info "Total packages in $name: $total_packages"
            
            # Check which packages are missing
            local missing_count=0
            while IFS= read -r package; do
                # Skip comments and empty lines
                [[ "$package" =~ ^#.*$ ]] && continue
                [[ -z "$package" ]] && continue
                
                # Extract package name (before >= or ==)
                pkg_name=$(echo "$package" | sed 's/[>=<].*//' | xargs)
                
                # Check if package is installed using UV
                if ! uv pip show "$pkg_name" &>/dev/null; then
                    missing_count=$((missing_count + 1))
                fi
            done < "$req_file"
            
            if [ $missing_count -gt 0 ]; then
                needs_install=true
            fi
            ;;
    esac
    
    if [ "$needs_install" = true ]; then
        print_warning "$name dependencies need to be installed"
        print_info "Installing dependencies with UV..."
        USER_SITE=$(python3 -m site --user-site)
        uv pip install -q -r "$req_file" --target "$USER_SITE"
        print_success "$name dependencies installed"
    else
        print_success "All $name dependencies are installed"
    fi
}

# Check Scout core dependencies
if [ -d "$PROJECT_ROOT/scout_data_discovery" ]; then
    check_and_install "$PROJECT_ROOT/scout_data_discovery" "Scout Core"
fi

# Check AI Functionality dependencies  
if [ -d "$PROJECT_ROOT/AI_Functionality" ]; then
    check_and_install "$PROJECT_ROOT/AI_Functionality" "AI Functionality"
fi

# Check backend dependencies
check_and_install "$PROJECT_ROOT/backend" "Backend"

# Check frontend dependencies
check_and_install "$PROJECT_ROOT/frontend" "Frontend"

# Clean up cache to save space
if [ "$1" = "--clean-cache" ] || [ "$2" = "--clean-cache" ]; then
    print_info "Cleaning pip cache..."
    pip cache purge 2>/dev/null || true
    print_info "Cleaning UV cache..."
    uv cache clean 2>/dev/null || true
    print_success "Package caches cleaned"
fi

# Check for outdated packages
if [ "$1" = "--check-outdated" ] || [ "$2" = "--check-outdated" ]; then
    print_info "Checking for outdated packages..."
    uv pip list --outdated
fi

echo
print_success "=== Dependency optimization complete ==="
