"""Interactive REPL for agent chat sessions."""
import json
from typing import List, Dict, Any, Optional
from agentsys import Agent, Response, Swarm
from openai import OpenAI

def safe_parse_json(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON string, returning empty dict if invalid.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as dict, or empty dict if invalid
    """
    try:
        return json.loads(json_str or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}

def process_streaming_response(response) -> Response:
    """Process and print streaming response with color output.
    
    Args:
        response: Streaming response from agent
        
    Returns:
        Final Response object with messages
    """
    content = ""
    last_sender = ""
    final_response = None

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f.get("name", "")
                if not name:
                    continue
                args = safe_parse_json(f.get("arguments", "{}"))
                arg_str = json.dumps(args).replace(":", "=")
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m({arg_str[1:-1]})")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "messages" in chunk:
            final_response = Response(messages=chunk["messages"])
            break

    return final_response or Response(messages=[])

def pretty_print_message(message: Dict[str, Any]) -> None:
    """Print a message with color formatting.
    
    Args:
        message: Message dictionary to print
    """
    if message["role"] != "assistant":
        return

    # Print agent name in blue
    print(f"\033[94m{message.get('sender', 'AI')}\033[0m:", end=" ")

    # Print response content
    if message.get("content"):
        print(message["content"])

    # Print tool calls in purple
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) > 1:
        print()
    for tool_call in tool_calls:
        f = tool_call["function"]
        name = f.get("name", "")
        if not name:
            continue
        args = safe_parse_json(f.get("arguments", "{}"))
        arg_str = json.dumps(args).replace(":", "=")
        print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

def run_agent_repl(
    agent: Agent,
    client: OpenAI,
    context_variables: Optional[Dict[str, Any]] = None,
    stream: bool = True,
    debug: bool = False
) -> None:
    """Run an interactive REPL session with an agent.
    
    Args:
        agent: The agent to chat with
        client: OpenAI client for API calls
        context_variables: Optional variables to pass to the agent
        stream: Whether to stream responses (default: True)
        debug: Whether to show debug information
    """
    print("\nChat with AI Assistant (type 'exit' to end)")
    print("I can help with greetings and calculations! For division, I'll call my calculator friend.\n")
    
    messages: List[Dict[str, Any]] = []
    swarm = Swarm(client=client)
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\033[90mUser\033[0m: ").strip()
            except EOFError:
                print("\nGoodbye! Chat ended.")
                break
                
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAI: Goodbye! Have a great day!")
                break
                
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            try:
                # Get response from agent via swarm
                response = swarm.run(
                    agent=agent,
                    messages=messages,
                    context_variables=context_variables or {},
                    stream=stream,
                    debug=debug,
                )
                
                if stream:
                    response = process_streaming_response(response)
                else:
                    for message in response.messages:
                        pretty_print_message(message)
                    
                messages.extend(response.messages)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                if debug:
                    raise
                print("Let's continue our conversation...")
                
    except KeyboardInterrupt:
        print("\nChat ended by user. Goodbye!")
