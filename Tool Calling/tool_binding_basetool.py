from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")


class Multiply(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers together."
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


class DivideInput(BaseModel):
    a: float = Field(required=True, description="The numerator to divide")
    b: float = Field(required=True, description="The denominator to divide")


class Divide(BaseTool):
    name: str = "divide"
    description: str = "Divide two numbers together."
    args_schema: Type[BaseModel] = DivideInput

    def _run(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Denominator cannot be zero.")
        return a / b


multiply_tool = Multiply()


llm_with_tool = llm.bind_tools([multiply_tool, Divide()])

response = llm_with_tool.invoke(
    [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 6 divide by 2?"),
    ]
)

print(response)
