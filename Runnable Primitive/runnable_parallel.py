from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-lite"
    )
    return model


model1 = get_gemini_model()
model2 = get_gemini_model()

template1 = PromptTemplate(
    template="Write a post for twitter on {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a post for LinkedIn on {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

chain1 = RunnableSequence(template1, model1, parser)
chain2 = RunnableSequence(template2, model2, parser)

final_chain = RunnableParallel({
    "tweet": chain1,
    "linkedin": chain2,
})

response = final_chain.invoke({"topic": "AI"})
print(response)
