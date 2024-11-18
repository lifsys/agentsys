"""Simple agent example with chat interface."""
from openai import OpenAI
from locksys import Locksys
import sys

from agentsys.repl import run_agent_repl
from agents import create_greeter_agent

def main():
    """Run the agent chat interface."""
    # Check if running in interactive mode
    if not sys.stdin.isatty():
        print("This script needs to run in an interactive terminal.")
        sys.exit(1)
        
    # Initialize OpenAI client with Locksys
    client = OpenAI(api_key=Locksys().item('OPEN-AI').key('Mamba').results())

    # Create agent and run REPL
    agent = create_greeter_agent()
    run_agent_repl(agent, client, stream=True, debug=False)

if __name__ == "__main__":
    main()
