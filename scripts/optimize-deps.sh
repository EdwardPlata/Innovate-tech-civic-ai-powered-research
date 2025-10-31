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

# Check if pip is up to date
print_info "Checking pip version..."
CURRENT_PIP_VERSION=$(pip --version | awk '{print $2}')
print_info "Current pip version: $CURRENT_PIP_VERSION"

# Upgrade pip if requested
if [ "$1" = "--upgrade-pip" ]; then
    print_info "Upgrading pip..."
    pip install --upgrade pip -q
    print_success "Pip upgraded"
fi

# Function to check and install requirements
check_and_install() {
    local req_file=$1
    local name=$2
    
    if [ ! -f "$req_file" ]; then
        print_warning "Requirements file not found: $req_file"
        return 1
    fi
    
    print_info "Checking $name dependencies..."
    
    # Count total packages
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
        
        # Check if package is installed
        if ! pip show "$pkg_name" &>/dev/null; then
            missing_count=$((missing_count + 1))
        fi
    done < "$req_file"
    
    if [ $missing_count -eq 0 ]; then
        print_success "All $name dependencies are installed"
    else
        print_warning "$missing_count packages need to be installed for $name"
        print_info "Installing missing dependencies..."
        pip install -r "$req_file" -q --no-cache-dir
        print_success "$name dependencies installed"
    fi
}

# Check Scout core dependencies
if [ -d "$PROJECT_ROOT/scout_data_discovery" ]; then
    SCOUT_REQ="$PROJECT_ROOT/scout_data_discovery/requirements.txt"
    check_and_install "$SCOUT_REQ" "Scout Core"
fi

# Check backend dependencies
BACKEND_REQ="$PROJECT_ROOT/backend/requirements.txt"
check_and_install "$BACKEND_REQ" "Backend"

# Check frontend dependencies
FRONTEND_REQ="$PROJECT_ROOT/frontend/requirements.txt"
check_and_install "$FRONTEND_REQ" "Frontend"

# Clean up pip cache to save space
if [ "$1" = "--clean-cache" ] || [ "$2" = "--clean-cache" ]; then
    print_info "Cleaning pip cache..."
    pip cache purge 2>/dev/null || true
    print_success "Pip cache cleaned"
fi

# Check for outdated packages
if [ "$1" = "--check-outdated" ] || [ "$2" = "--check-outdated" ]; then
    print_info "Checking for outdated packages..."
    pip list --outdated
fi

echo
print_success "=== Dependency optimization complete ==="
