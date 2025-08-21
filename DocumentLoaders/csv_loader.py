from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

loader = CSVLoader(file_path='demo.csv')

df = loader.load()

template = PromptTemplate(
    template="Answer the user query based on the content below: \n {content}\n\nquery:{query}",
    input_variables=["content", "query"],
)

parser = StrOutputParser()

chain = template | model | parser

query = "Draw me some 3 unique insights from the Data you have"

print(chain.invoke({"query":query, "content":df}))