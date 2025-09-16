#!/usr/bin/env python3
"""
Streamlit app runner for Scout Data Discovery Frontend

Simple script to start the Streamlit app.
The app will manage the backend automatically through the UI.
"""

import streamlit.web.cli as stcli
import sys
from pathlib import Path

def main():
    print("🚀 Starting Scout Data Discovery Frontend")
    print("=" * 50)
    print(f"🎨 Frontend will be available at: http://localhost:8501")
    print(f"🔄 Backend will be managed automatically through the UI")
    print("")

    # Check if app.py exists
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print(f"❌ app.py not found at: {app_path}")
        sys.exit(1)

    try:
        # Run Streamlit app
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.serverAddress=localhost",
            "--browser.serverPort=8501"
        ]

        stcli.main()

    except KeyboardInterrupt:
        print("\n⏹️  App stopped by user")
    except Exception as e:
        print(f"\n❌ App error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()