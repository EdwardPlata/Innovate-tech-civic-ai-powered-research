#!/bin/bash

# Scout Data Discovery Startup Script
# Starts both backend API server and frontend Streamlit app

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# PID files for process management
BACKEND_PID_FILE="$SCRIPT_DIR/backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/frontend.pid"

# Log files
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[SCOUT]${NC} $1"
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

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down Scout services..."

    # Stop backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_status "Stopping backend (PID: $BACKEND_PID)..."
            kill $BACKEND_PID
            wait $BACKEND_PID 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi

    # Stop frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_status "Stopping frontend (PID: $FRONTEND_PID)..."
            kill $FRONTEND_PID
            wait $FRONTEND_PID 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi

    print_success "Scout services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Function to start backend
start_backend() {
    print_status "Starting Scout backend..."

    # Check if backend directory exists
    if [ ! -d "$BACKEND_DIR" ]; then
        print_error "Backend directory not found: $BACKEND_DIR"
        exit 1
    fi

    # Check if backend port is already in use
    if check_port 8080; then
        print_warning "Port 8080 is already in use. Backend might already be running."
        print_status "Checking existing backend service..."
        if curl -s http://localhost:8080/api/health >/dev/null 2>&1; then
            print_success "Backend is already running and healthy!"
            return 0
        else
            print_error "Port 8080 is in use but backend is not responding. Please check for conflicts."
            exit 1
        fi
    fi

    # Start backend server
    cd "$BACKEND_DIR"

    # Check for requirements
    if [ -f "requirements.txt" ] && ! python -c "import uvicorn, fastapi" 2>/dev/null; then
        print_status "Installing backend dependencies..."
        USER_SITE=$(python3 -m site --user-site)
        uv pip install -r requirements.txt --target "$USER_SITE"
    fi

    # Start backend in background
    python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$BACKEND_PID_FILE"

    print_status "Backend starting (PID: $BACKEND_PID)..."

    # Wait for backend to be ready
    print_status "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/api/health >/dev/null 2>&1; then
            print_success "Backend is ready! (http://localhost:8080)"
            return 0
        fi
        sleep 1
        echo -n "."
    done

    print_error "Backend failed to start within 30 seconds"
    print_error "Check backend log: $BACKEND_LOG"
    exit 1
}

# Function to start frontend
start_frontend() {
    print_status "Starting Scout frontend..."

    # Check if frontend directory exists
    if [ ! -d "$FRONTEND_DIR" ]; then
        print_error "Frontend directory not found: $FRONTEND_DIR"
        exit 1
    fi

    # Check if frontend port is already in use
    if check_port 8501; then
        print_warning "Port 8501 is already in use. Frontend might already be running."
        print_status "You can access the frontend at: http://localhost:8501"
        return 0
    fi

    # Start frontend server
    cd "$FRONTEND_DIR"

    # Check for requirements
    if [ -f "requirements.txt" ] && ! python -c "import streamlit" 2>/dev/null; then
        print_status "Installing frontend dependencies..."
        USER_SITE=$(python3 -m site --user-site)
        uv pip install -r requirements.txt --target "$USER_SITE"
    fi

    # Start frontend in background
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$FRONTEND_PID_FILE"

    print_status "Frontend starting (PID: $FRONTEND_PID)..."

    # Wait for frontend to be ready
    print_status "Waiting for frontend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_success "Frontend is ready! (http://localhost:8501)"
            return 0
        fi
        sleep 1
        echo -n "."
    done

    print_warning "Frontend may still be starting. Check http://localhost:8501"
}

# Function to show status
show_status() {
    print_status "Scout Services Status:"
    echo

    # Backend status
    if check_port 8080; then
        if curl -s http://localhost:8080/api/health >/dev/null 2>&1; then
            print_success "✅ Backend: Running (http://localhost:8080)"
        else
            print_warning "⚠️  Backend: Port in use but not responding"
        fi
    else
        print_error "❌ Backend: Not running"
    fi

    # Frontend status
    if check_port 8501; then
        print_success "✅ Frontend: Running (http://localhost:8501)"
    else
        print_error "❌ Frontend: Not running"
    fi
    echo
}

# Main execution
print_success "🔍 Scout Data Discovery Startup"
print_status "Project directory: $SCRIPT_DIR"
echo

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --status, -s    Show service status"
    echo "  --stop          Stop running services"
    echo "  --help, -h      Show this help message"
    echo
    echo "Default: Start both backend and frontend services"
    exit 0
fi

# Check for status flag
if [ "$1" = "--status" ] || [ "$1" = "-s" ]; then
    show_status
    exit 0
fi

# Check for stop flag
if [ "$1" = "--stop" ]; then
    cleanup
    exit 0
fi

# Start services
print_status "Starting Scout services..."

# Start backend first
start_backend

# Start frontend
start_frontend

# Show final status
echo
print_success "🚀 Scout Data Discovery is ready!"
echo
print_success "📊 Frontend: http://localhost:8501"
print_success "🔧 Backend API: http://localhost:8080"
print_success "📋 API Docs: http://localhost:8080/docs"
echo
print_status "Press Ctrl+C to stop all services"
print_status "Logs: Backend ($BACKEND_LOG) | Frontend ($FRONTEND_LOG)"
echo

# Keep script running and monitor services
while true; do
    # Check if backend is still running
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            print_error "Backend process died unexpectedly!"
            print_error "Check backend log: $BACKEND_LOG"
            cleanup
            exit 1
        fi
    fi

    # Check if frontend is still running
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_error "Frontend process died unexpectedly!"
            print_error "Check frontend log: $FRONTEND_LOG"
            cleanup
            exit 1
        fi
    fi

    sleep 5
done