from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeEmbedding
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0,
    )
    return model


model = get_gemini_model()

prompt = PromptTemplate(
    template="Suggest a catchy Blog Title about {subject}",
    input_variables=["subject"],
)

#Chain
chain = LLMChain(llm=model, prompt=prompt)

topic = input("Enter a topic: ")
response = chain.invoke({"subject": topic})

print(response)
