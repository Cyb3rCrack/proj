from setuptools import setup, find_packages

setup(
    name="zypherus",
    version="0.2.0",
    description="Zypherus - Advanced AI system with persistent learning and engineering judgment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core ML/NLP
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.25.0",
        
        # Data/tables
        "pandas>=1.5.0",
        
        # Network
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        
        # Web framework & security
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "gunicorn>=20.0.0",
        "python-dotenv>=0.19.0",
        "Werkzeug>=2.2.0",
        "PyJWT>=2.6.0",
        
        # Data validation
        "pydantic>=1.10.0",
        
        # Processing
        "psutil>=5.9.0",
        
        # Logging
        "python-json-logger>=2.0.0",
        
        # Monitoring & error tracking
        "sentry-sdk>=1.5.0",
        
        # Testing
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "nlp": [
            "spacy>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "Open-ZypherusREPL=ace.cli.repl:main",
            "Invoke-ZypherusTest=ace.tests.test_core:main",
            "Start-ZypherusAPI=ace.api.server:run_server",
        ],
    },
)
