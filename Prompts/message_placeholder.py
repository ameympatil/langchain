from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

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

#chat template
chat_template = ChatPromptTemplate([
    ("system",'You are a helpful assitant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# load histiry
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# create prompt

prompt = chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund?'})
print(prompt)

response = model.invoke(prompt)

print(response.content)