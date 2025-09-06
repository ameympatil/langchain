from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001"
    )
    return model


prompt = PromptTemplate(
    template="Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["question", "context"],
)

retriver = WikipediaRetriever(top_k_results=2, lang="en")

model = get_gemini_model()
parser = StrOutputParser()


chain = {"context": retriver, "question": lambda x: x} | prompt | model | parser
response = chain.invoke("Who is the Prime Minister of India?")
print(response)
# Expected output: "The Prime Minister of India is Narendra Modi."
