from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


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

chat_history = []
chat_history.append(SystemMessage(content="Try to answer the question as best as you can and keep it simple and short."))


while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    else:
        response = model.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content))
        print(response.content)

print(chat_history)