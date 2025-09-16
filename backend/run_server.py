#!/usr/bin/env python3
"""
Server runner for Scout Data Discovery Backend

Simple script to start the FastAPI server with proper configuration.
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting Scout Data Discovery Backend API")
    print("=" * 50)

    # Check if scout_data_discovery is available
    scout_path = Path(__file__).parent.parent / "scout_data_discovery"
    if not scout_path.exists():
        print(f"❌ Scout Data Discovery not found at: {scout_path}")
        print("Please ensure the scout_data_discovery directory is in the parent folder.")
        sys.exit(1)

    print(f"✅ Scout Data Discovery found at: {scout_path}")
    print(f"📡 Starting server on http://localhost:8080")
    print(f"📖 API documentation available at: http://localhost:8080/docs")
    print(f"🔄 Auto-reload enabled for development")
    print("")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()