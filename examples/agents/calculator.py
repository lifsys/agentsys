"""Calculator agent and functions."""
from agentsys import Agent
import math

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)

def create_calculator_agent() -> Agent:
    """Create a calculator agent with mathematical capabilities."""
    return Agent(
        name="calculator",
        instructions="""You are a helpful calculator assistant that can perform mathematical operations.
        
        Available functions:
        1. add(a, b): Add two numbers
        2. subtract(a, b): Subtract b from a
        3. multiply(a, b): Multiply two numbers
        4. divide(a, b): Divide a by b (handles division by zero)
        5. square_root(x): Calculate square root of x (handles negative numbers)
        
        When users ask for calculations:
        1. Parse the numbers from their request
        2. Choose the appropriate function
        3. Handle errors gracefully
        4. Explain the calculation step by step
        5. Format the result clearly
        
        Always validate inputs and provide helpful error messages.""",
        functions=[add, subtract, multiply, divide, square_root]
    )

def run_example(user_input: str) -> None:
    """Run an example calculation with the given user input."""
    from openai import OpenAI
    from locksys import Locksys
    from agentsys import Swarm
    
    # Initialize OpenAI client with Locksys
    client = OpenAI(api_key=Locksys().item('OPEN-AI').key('Mamba').results())
    
    # Create calculator agent and swarm
    calculator = create_calculator_agent()
    swarm = Swarm(client=client)
    
    # Process user input
    messages = [{"role": "user", "content": user_input}]
    response = swarm.run(calculator, messages)
    
    # Print response
    for message in response.messages:
        if message["role"] == "assistant" and message.get("content"):
            print(f"\nCalculator: {message['content']}")

if __name__ == "__main__":
    # Example calculations
    examples = [
        "What is 15 plus 27?",
        "Can you multiply 8 by 6?",
        "What's 100 divided by 0?",  # Test error handling
        "What's the square root of 144?",
        "What's the square root of -9?"  # Test error handling
    ]
    
    print("Running calculator examples...")
    for example in examples:
        print(f"\nUser: {example}")
        try:
            run_example(example)
        except Exception as e:
            print(f"Error: {str(e)}")
    print("\nExamples complete.")
