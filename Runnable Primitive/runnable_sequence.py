from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"
    )
    return model


model = get_gemini_model()

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write me a joke on {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Explain the following joke in 1-2 lines: {response}",
    input_variables=["response"],
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)
response = chain.invoke({"topic": "India"}) 
print(response)
