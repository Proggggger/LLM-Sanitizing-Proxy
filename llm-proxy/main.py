"""Main entry point for running the LLM proxy server."""
import sys
from app.main import run_server

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_server(config_path)
