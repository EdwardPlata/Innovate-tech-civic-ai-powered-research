#!/bin/bash

# Scout Data Discovery - First-Time Setup Script
# Prepares the environment for the Scout platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_info() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_section() {
    echo
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo
}

# Display banner
clear
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════╗
║                                                   ║
║     Scout Data Discovery Platform - Setup        ║
║                                                   ║
║     First-Time Environment Configuration         ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

print_section "1. System Requirements Check"

# Check Python
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_info "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python $PYTHON_VERSION detected. Python 3.8+ required."
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check pip
print_info "Checking pip installation..."
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    print_error "pip is not installed"
    print_info "Installing pip..."
    python3 -m ensurepip --default-pip || curl https://bootstrap.pypa.io/get-pip.py | python3
fi
print_success "pip is available"

# Check disk space
print_info "Checking disk space..."
AVAILABLE_SPACE=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 2 ]; then
    print_warning "Low disk space detected: ${AVAILABLE_SPACE}GB available"
    print_warning "At least 2GB recommended"
else
    print_success "Disk space: ${AVAILABLE_SPACE}GB available"
fi

# Check memory
print_info "Checking system memory..."
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -g | awk 'NR==2 {print $2}')
    if [ "$TOTAL_MEM" -lt 4 ]; then
        print_warning "Low memory detected: ${TOTAL_MEM}GB total"
        print_warning "At least 4GB recommended for optimal performance"
    else
        print_success "System memory: ${TOTAL_MEM}GB total"
    fi
fi

print_section "2. Directory Structure Setup"

# Create necessary directories
print_info "Creating directory structure..."

mkdir -p "$SCRIPT_DIR/backend/cache/datasets"
mkdir -p "$SCRIPT_DIR/backend/cache/relationships"
mkdir -p "$SCRIPT_DIR/backend/cache/quality"
mkdir -p "$SCRIPT_DIR/logs"

print_success "Directory structure created"

print_section "3. Dependency Installation"

# Ask user for installation preference
echo
print_info "Ready to install dependencies. This may take a few minutes."
read -p "$(echo -e "${CYAN}Install dependencies now? [Y/n]: ${NC}")" -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_info "Installing dependencies..."
    
    # Run dependency optimizer
    if [ -f "$SCRIPT_DIR/scripts/optimize-deps.sh" ]; then
        bash "$SCRIPT_DIR/scripts/optimize-deps.sh" --upgrade-pip
    else
        # Fallback installation
        print_info "Installing backend dependencies..."
        cd "$SCRIPT_DIR/backend"
        pip install -q -r requirements.txt
        
        print_info "Installing frontend dependencies..."
        cd "$SCRIPT_DIR/frontend"
        pip install -q -r requirements.txt
        
        cd "$SCRIPT_DIR"
    fi
    
    print_success "Dependencies installed"
else
    print_warning "Skipping dependency installation"
    print_info "You can install them later with: ./scripts/optimize-deps.sh"
fi

print_section "4. Configuration"

# Check for existing configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    print_info "Configuration file (.env) already exists"
else
    print_info "No .env file found (optional)"
    print_info "Default configuration will be used"
fi

print_section "5. Setup Complete"

echo
print_success "Scout Data Discovery Platform is ready!"
echo
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  1. Start the platform:    ${GREEN}./run.sh${NC}"
echo -e "  2. Or use Docker:         ${GREEN}docker compose up -d${NC}"
echo -e "  3. Access frontend:       ${BLUE}http://localhost:8501${NC}"
echo -e "  4. Access backend API:    ${BLUE}http://localhost:8080${NC}"
echo
echo -e "${CYAN}Helpful Commands:${NC}"
echo -e "  • Check health:           ${GREEN}./scripts/health-check.sh${NC}"
echo -e "  • Optimize cache:         ${GREEN}./scripts/optimize-cache.sh${NC}"
echo -e "  • View documentation:     ${GREEN}cat README.md${NC}"
echo
print_info "For detailed deployment options, see: DEPLOYMENT_GUIDE.md"
echo

# Ask if user wants to start now
read -p "$(echo -e "${CYAN}Would you like to start Scout now? [Y/n]: ${NC}")" -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_info "Starting Scout Data Discovery Platform..."
    bash "$SCRIPT_DIR/run.sh"
else
    print_info "You can start Scout later with: ./run.sh"
fi
