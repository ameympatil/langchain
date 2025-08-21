from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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
prompt = PromptTemplate(
    template="Explain this poem: {poem}",
    input_variables=['poem']
)

loader = TextLoader('poem.txt')

# docs = loader.load()

docs = loader.lazy_load()

chain = prompt | model

print(chain.invoke(docs.page_content))