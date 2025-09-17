from langchain_core.tools import tool


# Step 1: Create a function
# Step 2: add type hints to the function parameters and return type
# Step 3 add tool decorator to the function
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


result = multiply.invoke({"a": 3, "b": 4})

print(result)  # should print 12
print(multiply.description)  # should print the function docstring
print(multiply.name)  # should print "multiply"
print(multiply.args)  # should print the function args schema