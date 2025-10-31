#!/bin/bash

# Health Check Script
# Verifies that Scout Data Discovery services are running properly

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
    echo -e "${BLUE}[HEALTH]${NC} $1"
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

# Configuration
BACKEND_URL="http://localhost:8080"
FRONTEND_URL="http://localhost:8501"
TIMEOUT=5

print_info "=== Scout Data Discovery Health Check ==="
echo

EXIT_CODE=0

# Check if curl is available
if ! command -v curl &> /dev/null; then
    print_error "curl is not installed. Please install curl for health checks."
    exit 1
fi

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local name=$2
    local expected_code=${3:-200}
    
    print_info "Checking $name..."
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$url" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "$expected_code" ] || [ "$HTTP_CODE" = "200" ]; then
        print_success "$name is healthy (HTTP $HTTP_CODE)"
        return 0
    elif [ "$HTTP_CODE" = "000" ]; then
        print_error "$name is not reachable"
        return 1
    else
        print_warning "$name returned HTTP $HTTP_CODE"
        return 1
    fi
}

# Check backend health endpoint
if check_endpoint "$BACKEND_URL/api/health" "Backend API"; then
    # Get backend version/info if available
    BACKEND_INFO=$(curl -s --connect-timeout $TIMEOUT "$BACKEND_URL/api/health" 2>/dev/null || echo "{}")
    if [ "$BACKEND_INFO" != "{}" ]; then
        print_info "Backend details: $BACKEND_INFO"
    fi
else
    EXIT_CODE=1
fi

echo

# Check backend API docs
if check_endpoint "$BACKEND_URL/docs" "Backend API Docs"; then
    print_info "API documentation available at: $BACKEND_URL/docs"
else
    print_warning "API documentation may not be accessible"
fi

echo

# Check frontend health
if check_endpoint "$FRONTEND_URL/_stcore/health" "Frontend App"; then
    print_info "Frontend UI available at: $FRONTEND_URL"
else
    EXIT_CODE=1
fi

echo

# Check if processes are running (if PID files exist)
BACKEND_PID_FILE="$PROJECT_ROOT/backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/frontend.pid"

if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend process running (PID: $BACKEND_PID)"
    else
        print_error "Backend PID file exists but process is not running"
        EXIT_CODE=1
    fi
else
    print_info "No backend PID file found (service may be running independently)"
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_success "Frontend process running (PID: $FRONTEND_PID)"
    else
        print_error "Frontend PID file exists but process is not running"
        EXIT_CODE=1
    fi
else
    print_info "No frontend PID file found (service may be running independently)"
fi

echo

# Check disk space
print_info "Checking disk space..."
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    print_error "Disk usage is critical: ${DISK_USAGE}%"
    EXIT_CODE=1
elif [ "$DISK_USAGE" -gt 80 ]; then
    print_warning "Disk usage is high: ${DISK_USAGE}%"
else
    print_success "Disk usage is acceptable: ${DISK_USAGE}%"
fi

echo

# Check log files
print_info "Checking log files..."
BACKEND_LOG="$PROJECT_ROOT/backend.log"
FRONTEND_LOG="$PROJECT_ROOT/frontend.log"

if [ -f "$BACKEND_LOG" ]; then
    LOG_SIZE=$(du -h "$BACKEND_LOG" | cut -f1)
    ERROR_COUNT=$(grep -i "error" "$BACKEND_LOG" | wc -l)
    print_info "Backend log: $LOG_SIZE (${ERROR_COUNT} errors)"
    
    if [ "$ERROR_COUNT" -gt 10 ]; then
        print_warning "High number of errors in backend log"
        if [ "$1" = "--verbose" ] || [ "$1" = "-v" ]; then
            print_info "Recent errors:"
            tail -20 "$BACKEND_LOG" | grep -i "error" | tail -5
        fi
    fi
fi

if [ -f "$FRONTEND_LOG" ]; then
    LOG_SIZE=$(du -h "$FRONTEND_LOG" | cut -f1)
    ERROR_COUNT=$(grep -i "error" "$FRONTEND_LOG" | wc -l)
    print_info "Frontend log: $LOG_SIZE (${ERROR_COUNT} errors)"
    
    if [ "$ERROR_COUNT" -gt 10 ]; then
        print_warning "High number of errors in frontend log"
        if [ "$1" = "--verbose" ] || [ "$1" = "-v" ]; then
            print_info "Recent errors:"
            tail -20 "$FRONTEND_LOG" | grep -i "error" | tail -5
        fi
    fi
fi

echo

# Check Python dependencies
if [ "$1" = "--check-deps" ]; then
    print_info "Checking critical dependencies..."
    
    MISSING_DEPS=""
    for dep in fastapi uvicorn streamlit pandas requests; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            MISSING_DEPS="$MISSING_DEPS $dep"
            print_error "Missing dependency: $dep"
            EXIT_CODE=1
        else
            print_success "Dependency installed: $dep"
        fi
    done
    
    if [ -z "$MISSING_DEPS" ]; then
        print_success "All critical dependencies are installed"
    else
        print_error "Missing dependencies:$MISSING_DEPS"
    fi
    echo
fi

# Overall status
echo "════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Overall Status: HEALTHY"
else
    print_error "Overall Status: UNHEALTHY"
    print_info "Some services may need attention"
fi
echo "════════════════════════════════════════"

exit $EXIT_CODE
