"""Greeter agent and functions."""
from agentsys import Agent
from .calculator import (
    add, subtract, multiply, divide,
    square_root, create_calculator_agent
)

def greet(name: str) -> str:
    """Simple greeting function."""
    return f"Hello, {name}!"

def create_greeter_agent() -> Agent:
    """Create a greeter agent that can handle math and conversation."""
    return Agent(
        name="assistant",
        instructions="""You are a friendly assistant who can help with conversations and calculations. You can:
        1. Greet users when they introduce themselves using the greet function
        2. Perform mathematical calculations:
           - add(a, b): Add two numbers
           - subtract(a, b): Subtract b from a
           - multiply(a, b): Multiply two numbers
           - divide(a, b): Divide a by b (handles division by zero)
           - square_root(x): Calculate square root (handles negative numbers)
        3. Have casual conversations about any topic
        
        When users ask for calculations:
        1. Parse the numbers from their request carefully
        2. Choose the appropriate function
        3. Handle errors gracefully (e.g. division by zero)
        4. Explain the calculation step by step
        5. Format the result clearly
        
        For division:
        1. Extract the dividend (a) and divisor (b)
        2. Use the divide(a, b) function
        3. Handle division by zero gracefully
        4. Show the calculation steps
        
        Always be helpful and maintain a natural conversation flow.""",
        model="gpt-4",
        functions=[greet, add, subtract, multiply, divide, square_root]
    )
