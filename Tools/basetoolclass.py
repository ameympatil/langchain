from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers."

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
    async def _arun(self, a: int, b: int) -> int:
        raise NotImplementedError("MultiplyTool does not support async")
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)  # should print 12
