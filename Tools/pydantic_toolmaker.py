from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers together.",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)  # should print 12
