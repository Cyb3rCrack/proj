#!/usr/bin/env python3
"""
Build script for production obfuscation and deployment.
Compiles sensitive modules to bytecode to protect proprietary code.
"""

import os
import py_compile
import shutil
import json
from pathlib import Path

# Modules to obfuscate (compile to bytecode and remove source)
SENSITIVE_MODULES = [
    "Zypherus/core",
    "Zypherus/inference", 
    "Zypherus/memory",
    "Zypherus/extraction",
    "Zypherus/ingestion",
]

def compile_module_to_bytecode(module_path):
    """Compile a Python module to bytecode and remove source."""
    module_dir = Path(module_path)
    
    if not module_dir.exists():
        print(f"Warning: {module_path} not found")
        return
    
    print(f"Compiling {module_path} to bytecode...")
    
    for py_file in module_dir.rglob("*.py"):
        try:
            # Compile to bytecode
            pyc_file = py_file.with_suffix(".pyc")
            py_compile.compile(str(py_file), str(pyc_file), doraise=True)
            print(f"  ✓ Compiled {py_file.name}")
            
            # Keep source for now (set REMOVE_SOURCE=1 to delete after compilation)
            if os.getenv("REMOVE_SOURCE") == "1":
                os.remove(py_file)
                print(f"  ✗ Removed source {py_file.name}")
        except Exception as e:
            print(f"  ✗ Error compiling {py_file}: {e}")

def minify_json_config():
    """Minify JSON configuration files."""
    config_files = [
        "config/zypherus_config.json",
        "claims.json",
        "definitions.json",
        "ontology.json",
        "relationships.json",
    ]
    
    print("\nMinifying configuration files...")
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                # Write back without whitespace
                with open(config_file, 'w') as f:
                    json.dump(data, f, separators=(',', ':'))
                print(f"  ✓ Minified {config_file}")
            except Exception as e:
                print(f"  ✗ Error minifying {config_file}: {e}")

def create_deployment_bundle():
    """Create a clean deployment bundle."""
    print("\nPreparing deployment bundle...")
    
    # Files to exclude from deployment
    exclude_patterns = [
        "*.py",  # Source code (use compiled .pyc instead)
        ".*",     # Hidden files
        "__pycache__",
        ".pytest_cache",
        "*.egg-info",
        ".git",
        ".env",   # Never commit .env files
        "scripts/",  # Remove collection scripts
        "training_data/",  # Exclude training data
        "*.md",  # Remove documentation
    ]
    
    print("  Creating .renderignore...")
    with open(".renderignore", "w") as f:
        for pattern in exclude_patterns:
            f.write(f"{pattern}\n")
    
    print("  ✓ Deployment bundle prepared")

def main():
    print("=" * 60)
    print("ZYPHERUS PRODUCTION BUILD & OBFUSCATION")
    print("=" * 60)
    
    # Compile sensitive modules
    for module in SENSITIVE_MODULES:
        compile_module_to_bytecode(module)
    
    # Minify configs
    minify_json_config()
    
    # Create deployment bundle
    create_deployment_bundle()
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print("\nDeployment files ready:")
    print("  • Procfile - deployment startup config")
    print("  • requirements-prod.txt - dependencies")
    print("  • .renderignore - files to exclude")
    print("  • .env.example - environment template")
    print("\nNext steps:")
    print("  1. Set environment variables on Render")
    print("  2. Connect Git repository to Render")
    print("  3. Deploy will auto-trigger on git push")

if __name__ == "__main__":
    main()
