from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=10,
)

response = model.invoke("What is the capital of India?")
print(response.content)
