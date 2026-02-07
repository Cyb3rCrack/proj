"""
Installation script for Zypherus with all dependencies.
Run: python install_dependencies.py
"""

import subprocess
import sys


def install_packages():
    """Install all required packages."""
    
    packages = [
        # Core
        ("numpy", "1.24.0"),
        ("sentence-transformers", "2.2.0"),
        ("faiss-cpu", "1.7.0"),
        ("torch", "2.0.0"),
        ("transformers", "4.25.0"),
        ("pandas", "1.5.0"),
        
        # Web & API
        ("flask", "2.0.0"),
        ("flask-cors", "4.0.0"),
        ("requests", "2.28.0"),
        ("gunicorn", "20.0.0"),
        
        # Security & Auth
        ("PyJWT", "2.6.0"),
        ("python-dotenv", "0.19.0"),
        
        # Validation
        ("pydantic", "1.10.0"),
        
        # Monitoring
        ("sentry-sdk", "1.5.0"),
    ]
    
    print("Installing Zypherus dependencies...")
    print("=" * 50)
    
    failed = []
    
    for package, version in packages:
        try:
            print(f"Installing {package}=={version}...", end=" ")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", f"{package}=={version}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓")
        except subprocess.CalledProcessError:
            print("✗")
            failed.append(package)
    
    print("=" * 50)
    
    if failed:
        print(f"\nFailed to install: {', '.join(failed)}")
        print("Try installing manually:")
        for pkg in failed:
            print(f"  pip install {pkg}")
        sys.exit(1)
    else:
        print("\n✓ All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Configure your settings")
        print("3. Run: python -m Zypherus.cli.cli_user server start")


if __name__ == "__main__":
    install_packages()
