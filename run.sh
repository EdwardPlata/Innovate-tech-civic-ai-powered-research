#!/bin/bash

# Scout Data Discovery - Main Run Script
# Simplified deployment script that sets up and runs both backend and frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Display banner
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════╗"
echo "║   Scout Data Discovery Platform - Quick Start    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python installation
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check UV installation
print_info "Checking UV package manager..."
if ! command -v uv &> /dev/null; then
    print_warning "UV is not installed. Installing UV..."
    pip install uv
    print_success "UV installed"
else
    print_success "UV is available"
fi

# Check if optimization scripts should be run
if [ "$1" = "--optimize" ] || [ "$1" = "-o" ]; then
    print_info "Running optimization scripts..."
    
    # Run dependency optimizer
    if [ -f "$SCRIPT_DIR/scripts/optimize-deps.sh" ]; then
        bash "$SCRIPT_DIR/scripts/optimize-deps.sh"
    fi
    
    # Run cache optimizer
    if [ -f "$SCRIPT_DIR/scripts/optimize-cache.sh" ]; then
        bash "$SCRIPT_DIR/scripts/optimize-cache.sh"
    fi
fi

# Setup phase
print_info "Setting up Scout Data Discovery Platform..."

# Check and install backend dependencies
print_info "Checking backend dependencies..."
cd "$SCRIPT_DIR/backend"
if [ -f "requirements.txt" ]; then
    if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
        print_warning "Backend dependencies not fully installed. Installing..."
        USER_SITE=$(python3 -m site --user-site)
        uv pip install -q -r requirements.txt --target "$USER_SITE"
        print_success "Backend dependencies installed"
    else
        print_success "Backend dependencies already installed"
    fi
fi

# Check and install frontend dependencies
print_info "Checking frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
if [ -f "requirements.txt" ]; then
    if ! python3 -c "import streamlit" 2>/dev/null; then
        print_warning "Frontend dependencies not fully installed. Installing..."
        USER_SITE=$(python3 -m site --user-site)
        uv pip install -q -r requirements.txt --target "$USER_SITE"
        print_success "Frontend dependencies installed"
    else
        print_success "Frontend dependencies already installed"
    fi
fi

# Return to script directory
cd "$SCRIPT_DIR"

# Start services using the existing start_scout.sh script
print_info "Starting Scout services..."
echo

if [ -f "$SCRIPT_DIR/start_scout.sh" ]; then
    bash "$SCRIPT_DIR/start_scout.sh" "$@"
else
    print_error "start_scout.sh not found!"
    print_info "Please ensure start_scout.sh exists in $SCRIPT_DIR"
    exit 1
fi
