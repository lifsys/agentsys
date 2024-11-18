from agentsys import Agent, Swarm
import openai

def reverse_text(text: str) -> str:
    """Reverse the input text."""
    return text[::-1]

def uppercase_text(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

# Create an agent with text processing capabilities
agent = Agent(
    name="text_processor",
    instructions="""You are a helpful assistant that can process text.
    When asked to reverse text, use the reverse_text function.
    When asked to make text uppercase, use the uppercase_text function.
    Always explain what you did with the text.""",
    model="gpt-4",
    functions=[reverse_text, uppercase_text]
)

# Create a swarm to manage the agent
swarm = Swarm()

# Example conversation
messages = [
    {"role": "user", "content": "Please reverse the text 'hello world'"}
]

# Run the conversation
response = swarm.run(agent, messages)

# Print the conversation
print("\nConversation:")
for message in response.messages:
    role = message["role"]
    content = message.get("content", "")
    if role == "user":
        print(f"\nUser: {content}")
    elif role == "assistant":
        print(f"\nAssistant: {content}")
    elif role == "tool":
        print(f"\nTool Output: {content}")

# Try another example
messages = [
    {"role": "user", "content": "Make 'hello world' uppercase please"}
]

response = swarm.run(agent, messages)

print("\nConversation:")
for message in response.messages:
    role = message["role"]
    content = message.get("content", "")
    if role == "user":
        print(f"\nUser: {content}")
    elif role == "assistant":
        print(f"\nAssistant: {content}")
    elif role == "tool":
        print(f"\nTool Output: {content}")
