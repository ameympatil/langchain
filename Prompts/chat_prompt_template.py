from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

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


# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} expert."),
#     HumanMessage(content="Explain in simple terms, what is {topic}?")
# ],)

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert."),
    ('human',"Explain in simple terms, what is {topic}?")
])

prompt_template = PromptTemplate(
    template="What is capital of {country}?"
)

prompt2 = prompt_template.invoke({"country":"India"})

prompt = chat_template.invoke({"domain":"investment","topic":"Mutual funds"})
print(prompt)


response = model.invoke(prompt)
print(response.content)