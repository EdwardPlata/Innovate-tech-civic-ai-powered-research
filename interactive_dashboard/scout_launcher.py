"""
Scout Integration Launcher
Launches the interactive dashboard with Scout Data Discovery integration
"""
import os
import sys
import logging
from pathlib import Path

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'dash', 'plotly', 'pandas', 'numpy', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_scout_api():
    """Check if Scout API is available"""
    import requests
    from scout_config import SCOUT_API_URL
    
    try:
        response = requests.get(f"{SCOUT_API_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Scout API is available at {SCOUT_API_URL}")
            return True
        else:
            logger.warning(f"Scout API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Scout API not available: {e}")
        return False

def run_demo():
    """Run the component demonstration"""
    print("Running Scout Integration Demo...")
    
    try:
        from examples.complete_demo import run_complete_demo
        run_complete_demo()
        return True
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

def launch_dashboard():
    """Launch the main dashboard application"""
    from scout_config import get_dashboard_config
    
    config = get_dashboard_config()
    
    print(f"Launching Interactive Dashboard...")
    print(f"URL: http://{config['host']}:{config['port']}")
    print("Press Ctrl+C to stop the server")
    
    # Import and run the existing dashboard app
    try:
        import app
        if hasattr(app, 'app'):
            app.app.run_server(
                debug=config['debug'],
                host=config['host'],
                port=config['port']
            )
        else:
            logger.error("Could not find dashboard app instance")
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")

def main():
    """Main launcher function"""
    print("Scout Data Discovery - Interactive Dashboard")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Scout API availability
    scout_available = check_scout_api()
    if not scout_available:
        print("Warning: Scout API not available. Some features may not work.")
    
    # Get user choice
    print("\nOptions:")
    print("1. Run component demonstration")
    print("2. Launch interactive dashboard")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_demo()
                break
            elif choice == "2":
                launch_dashboard()
                break
            elif choice == "3":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()