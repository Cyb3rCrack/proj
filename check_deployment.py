#!/usr/bin/env python3
"""
Pre-deployment verification script for Railway.
Run this before pushing to ensure safe deployment.
"""

import os
import sys
import json
from pathlib import Path

class DeploymentChecker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name, condition, severity="error"):
        if condition:
            print(f"  [OK] {name}")
            self.passed += 1
        else:
            status = "[ERROR]" if severity == "error" else "[WARN]"
            print(f"  {status}: {name}")
            if severity == "error":
                self.failed += 1
            else:
                self.warnings += 1

    def section(self, title):
        print(f"\n{title}")
        print("=" * 60)

    def report(self):
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {self.passed} passed, {self.warnings} warnings, {self.failed} errors")
        print(f"{'=' * 60}")
        return self.failed == 0

def main():
    print("=" * 60)
    print("ZYPHERUS RENDER DEPLOYMENT CHECKER")
    print("=" * 60)

    checker = DeploymentChecker()

    # Check Git status
    checker.section("Git Repository")
    checker.check("Git repository initialized", os.path.exists(".git"))
    checker.check("Repository has commits", os.path.exists(".git/logs/HEAD"))
    
    # Check required files
    checker.section("Required Deployment Files")
    checker.check("Procfile exists", os.path.exists("Procfile"))
    checker.check("Dockerfile exists", os.path.exists("Dockerfile"))
    checker.check(".renderignore exists", os.path.exists(".renderignore"))
    checker.check("requirements-prod.txt exists", os.path.exists("requirements-prod.txt"))
    checker.check("setup.py exists", os.path.exists("setup.py"))
    checker.check("wsgi.py exists", os.path.exists("wsgi.py"))
    checker.check(".env.example exists", os.path.exists(".env.example"))

    # Check environment
    checker.section("Environment Configuration")
    
    env_exists = os.path.exists(".env")
    checker.check(".env exists locally", env_exists, "warning")
    
    if env_exists:
        with open(".env") as f:
            env_content = f.read()
            checker.check(".env not committed to Git", ".env" in open(".gitignore").read(), "warning")
            checker.check("FLASK_SECRET_KEY set", "FLASK_SECRET_KEY=" in env_content)
            checker.check("DEBUG=false", "DEBUG=false" in env_content)
            checker.check("FLASK_ENV=production", "FLASK_ENV=production" in env_content)

    # Check .gitignore
    checker.section("Git Ignore Configuration")
    if os.path.exists(".gitignore"):
        gitignore = open(".gitignore").read()
        checker.check(".env in .gitignore", ".env" in gitignore)
        checker.check(".venv in .gitignore", ".venv" in gitignore)
        checker.check("__pycache__ in .gitignore", "__pycache__" in gitignore)
    else:
        checker.check(".gitignore exists", False)

    # Check sensitive files
    checker.section("Security - No Sensitive Files")
    checker.check("No unencrypted secrets in code", 
                 "password=" not in open("wsgi.py").read().lower() and
                 "secret=" not in open("wsgi.py").read().lower(),
                 "error")
    
    checker.check("No API keys in source", 
                 "sk-" not in open("wsgi.py").read() and
                 "token=" not in open("wsgi.py").read().lower(),
                 "error")

    # Check build process
    checker.section("Build Configuration")
    checker.check("build_release.py is executable", 
                 os.access("build_release.py", os.X_OK), "warning")
    
    try:
        with open("build_release.py") as f:
            build_content = f.read()
            checker.check("Bytecode compilation configured", 
                         "py_compile" in build_content)
            checker.check("JSON minification configured",
                         "json.dump" in build_content)
    except:
        checker.check("build_release.py readable", False)

    # Check dependencies
    checker.section("Dependencies")
    try:
        with open("requirements-prod.txt") as f:
            reqs = f.read()
            checker.check("gunicorn in requirements", "gunicorn" in reqs)
            checker.check("flask in requirements", "flask" in reqs)
            # Check for uncommented pytest (dev dependencies should be commented out)
            uncommented_pytest = any("pytest" in line and not line.strip().startswith("#") 
                                    for line in reqs.split("\n"))
            checker.check("No dev dependencies in prod requirements", not uncommented_pytest, "warning")
    except:
        checker.check("requirements-prod.txt readable", False)

    # Check project structure
    checker.section("Project Structure")
    checker.check("Zypherus package directory exists", os.path.isdir("Zypherus"))
    checker.check("main.py exists", os.path.exists("main.py"))
    checker.check("setup.py exists", os.path.exists("setup.py"))

    # Final report
    success = checker.report()

    if success:
        print("\n[SUCCESS] ALL CHECKS PASSED - Ready for deployment!")
        print("\nNext steps:")
        print("  1. Commit changes: git add . && git commit -m 'Prepare for Render deployment'")
        print("  2. Push to GitHub: git push origin main")
        print("  3. Render will auto-deploy")
        print("  4. Monitor at: https://dashboard.render.com")
        return 0
    else:
        print("\n[FAILED] DEPLOYMENT CHECK FAILED")
        print("\nPlease fix the errors above before deploying.")
        print("\nCommon issues:")
        print("  * Missing .env file - copy .env.example and fill in values")
        print("  * .env not in .gitignore - add to .gitignore immediately")
        print("  * No FLASK_SECRET_KEY - generate: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        return 1

if __name__ == "__main__":
    sys.exit(main())
