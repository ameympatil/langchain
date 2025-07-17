from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()

def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-001",
        temperature=0,
    )
    return model

model = get_gemini_model()

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
    HumanMessage(content="Tell me about langchain in one line")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)