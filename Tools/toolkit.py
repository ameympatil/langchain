from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int: 
    """Add two numbers."""
    return a + b

class Toolkit:
    """A toolkit for basic arithmetic operations."""

    def get_tools(self):
        """Get the list of tools in the toolkit."""
        return [multiply, add]
    
toolkit = Toolkit()
tools = toolkit.get_tools()
for t in tools:
    print(f"Tool name: {t.name}, Description: {t.description}")