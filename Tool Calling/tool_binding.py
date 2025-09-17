from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")


class DivideInput(BaseModel):
    a: float = Field(required=True, description="The numerator to divide")
    b: float = Field(required=True, description="The denominator to divide")


def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"Multiplying {a} and {b}")
    return a * b


def divide_func(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    print(f"Dividing {a} by {b}")
    return a / b


multiply = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers together.",
    args_schema=MultiplyInput,
)

divide = StructuredTool.from_function(
    func=divide_func,
    name="divide",
    description="Divide two numbers together.",
    args_schema=DivideInput,
)


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0,
        tools=[multiply],
    )
    return model


model = get_gemini_model()

model_with_tools = model.bind_tools([multiply, divide])

chat = [
    SystemMessage(content="You are a helpful assistant who helps in maths."),
    HumanMessage(content="What is 7 multiply with 6 divide by 2?"),
]

response_with_tools = model_with_tools.invoke(chat)

response = multiply.invoke(response_with_tools.tool_calls[0])

chat.append(response_with_tools)
chat.append(response)

response_with_tools = model_with_tools.invoke(chat)

print(response_with_tools)

print("Final Answer:", response_with_tools.content)
