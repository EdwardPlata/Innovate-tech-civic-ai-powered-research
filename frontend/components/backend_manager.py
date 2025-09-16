"""
Backend Manager for Streamlit Frontend

Manages the FastAPI backend server directly from the frontend application.
"""

import subprocess
import sys
import time
import requests
import streamlit as st
from pathlib import Path
import threading
import atexit
import signal
import os

class BackendManager:
    """Manages the FastAPI backend server lifecycle"""

    def __init__(self):
        self.process = None
        self.backend_url = "http://localhost:8080"
        self.backend_path = Path(__file__).parent.parent.parent / "backend"
        self.is_running = False

        # Register cleanup
        atexit.register(self.stop_backend)

    def check_backend_health(self):
        """Check if backend is running and healthy"""
        try:
            # Try the health endpoint first (more reliable)
            response = requests.get(f"{self.backend_url}/api/health", timeout=5)
            if response.status_code == 200:
                return True

            # Fallback to root endpoint
            response = requests.get(f"{self.backend_url}/", timeout=5)
            return response.status_code == 200

        except Exception as e:
            # Debug: log the actual error
            print(f"Backend health check failed: {e}")
            return False

    def start_backend(self):
        """Start the backend server if not already running"""
        if self.is_running:
            return True

        # Check if backend is already running externally
        if self.check_backend_health():
            self.is_running = True
            return True

        try:
            # Start backend process
            backend_main = self.backend_path / "main.py"
            if not backend_main.exists():
                st.error(f"Backend main.py not found at {backend_main}")
                return False

            # Change to backend directory
            original_dir = os.getcwd()
            os.chdir(self.backend_path)

            # Start uvicorn server
            cmd = [
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", "8080",
                "--reload"
            ]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            # Restore original directory
            os.chdir(original_dir)

            # Wait for server to be ready
            max_attempts = 15
            for attempt in range(max_attempts):
                if self.check_backend_health():
                    self.is_running = True
                    return True
                time.sleep(2)

            # If we get here, startup failed
            self.stop_backend()
            return False

        except Exception as e:
            st.error(f"Failed to start backend: {e}")
            return False

    def stop_backend(self):
        """Stop the backend server"""
        if self.process:
            try:
                # Kill process group to ensure all child processes are terminated
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()

                self.process.wait(timeout=5)
            except:
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                except:
                    pass

            self.process = None

        self.is_running = False

    def get_status(self):
        """Get backend status information"""
        is_healthy = self.check_backend_health()

        return {
            "is_running": is_healthy,
            "url": self.backend_url,
            "process_active": self.process is not None and self.process.poll() is None,
            "health_check": is_healthy
        }

# Global backend manager instance
_backend_manager = None

def get_backend_manager():
    """Get the global backend manager instance"""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = BackendManager()
    return _backend_manager