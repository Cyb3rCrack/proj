"""User-friendly CLI interface for Zypherus."""

import click
import json
from pathlib import Path
from Zypherus.api.server import ZypherusAPIClient


@click.group()
def cli():
    """Zypherus - Advanced Reasoning System with Persistent Learning."""
    pass


@cli.group()
def server():
    """Server commands."""
    pass


@cli.group()
def client():
    """Client commands."""
    pass


@server.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def start(host, port, debug):
    """Start the Zypherus server."""
    try:
        from Zypherus.core.ace import ACE
        
        click.echo("Starting Zypherus server...")
        ace = ACE()
        
        from Zypherus.api.server import ZypherusAPIServer
        app_server = ZypherusAPIServer(ace, host=host, port=port)
        
        click.echo(f"Server running on http://{host}:{port}")
        click.echo(f"API Documentation: http://{host}:{port}/api/docs")
        click.echo("Press Ctrl+C to stop")
        
        app_server.run(debug=debug)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@client.command()
@click.option("--endpoint", default="http://localhost:8000", help="API endpoint")
def status(endpoint):
    """Check server status."""
    try:
        api_client = ZypherusAPIClient(endpoint)
        
        if api_client.health_check():
            click.echo("✓ Server is healthy")
            status_data = api_client.get_status()
            click.echo("\nSystem Status:")
            for key, value in status_data["data"].items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("✗ Server is not responding", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@client.command()
@click.option("--endpoint", default="http://localhost:8000", help="API endpoint")
@click.option("--file", type=click.File("r"), help="File to ingest")
@click.option("--text", help="Text to ingest")
@click.option("--source", default="cli", help="Source identifier")
def ingest(endpoint, file, text, source):
    """Ingest document or text."""
    try:
        api_client = ZypherusAPIClient(endpoint)
        
        if file:
            content = file.read()
            click.echo(f"Ingesting from file: {file.name}")
        elif text:
            content = text
            click.echo(f"Ingesting text ({len(content)} characters)")
        else:
            click.echo("Error: Provide either --file or --text", err=True)
            return
        
        result = api_client.ingest(content, source=source)
        
        if result["success"]:
            click.echo(f"✓ {result['data']['message']}")
        else:
            click.echo(f"✗ Error: {result['error']}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@client.command()
@click.option("--endpoint", default="http://localhost:8000", help="API endpoint")
@click.argument("query")
def ask(endpoint, query):
    """Ask a question."""
    try:
        api_client = ZypherusAPIClient(endpoint)
        
        click.echo(f"Question: {query}")
        click.echo("Thinking...", nl=False)
        
        result = api_client.answer(query)
        
        if result["success"]:
            data = result["data"]
            click.echo("\b\b")  # Remove "thinking..."
            click.echo(f"\nAnswer: {data.get('answer', 'No answer found')}")
            click.echo(f"Confidence: {data.get('confidence', 0):.1%}")
        else:
            click.echo(f"\n✗ Error: {result['error']}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@client.command()
@click.option("--endpoint", default="http://localhost:8000", help="API endpoint")
@click.option("--limit", default=5, help="Max results")
@click.argument("query")
def search(endpoint, limit, query):
    """Search knowledge base."""
    try:
        api_client = ZypherusAPIClient(endpoint)
        
        click.echo(f"Searching for: {query}")
        result = api_client.search(query, limit=limit)
        
        if result["success"]:
            results = result["data"]["results"]
            if results:
                click.echo(f"Found {len(results)} results:\n")
                for i, item in enumerate(results, 1):
                    text = item.get("text", "")[:100]
                    source = item.get("source", "unknown")
                    click.echo(f"{i}. {text}...")
                    click.echo(f"   Source: {source}\n")
            else:
                click.echo("No results found")
        else:
            click.echo(f"✗ Error: {result['error']}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@client.command()
@click.option("--endpoint", default="http://localhost:8000", help="API endpoint")
def info(endpoint):
    """Get system information."""
    try:
        api_client = ZypherusAPIClient(endpoint)
        
        click.echo("Zypherus System Information\n" + "=" * 40)
        
        # Status
        status_data = api_client.get_status()
        click.echo("\nMemory & Knowledge Base:")
        for key, value in status_data["data"].items():
            if key != "timestamp":
                click.echo(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Memory
        memory_data = api_client.get_memory()
        click.echo("\nMemory System:")
        for key, value in memory_data["data"].items():
            if key != "timestamp":
                if key == "sources" and isinstance(value, list):
                    click.echo(f"  Sources: {', '.join(value)}")
                else:
                    click.echo(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Beliefs
        beliefs_data = api_client.get_beliefs()
        click.echo(f"\nBeliefs: {beliefs_data['data'].get('total_beliefs', 0)} total")
        
        # Concepts
        concepts_data = api_client.get_concepts()
        click.echo(f"Concepts: {concepts_data['data'].get('total_concepts', 0)} entities")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@server.command()
def init():
    """Initialize Zypherus directories and config."""
    try:
        dirs = [
            "data/memory",
            "data/knowledge",
            "data/dialogues",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            click.echo(f"✓ Created {dir_path}")
        
        # Create example .env if not exists
        env_path = Path(".env")
        if not env_path.exists():
            click.echo("Note: Copy .env.example to .env and configure for your environment")
        
        click.echo("\n✓ Zypherus initialized successfully")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
def version():
    """Show version."""
    click.echo("Zypherus v0.2.0")


if __name__ == "__main__":
    cli()
