from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def wordcounter(text):
    return len(text.split())


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"
    )
    return model


model = get_gemini_model()

parser = StrOutputParser()

report_generation_prompt = PromptTemplate(
    template="Write me detailed report on {topic}", input_variables=["topic"]
)

summarize_report_prompt = PromptTemplate(
    template="Summarize this report: {report}", input_variables=["report"]
)

report_gen_chain = RunnableSequence(report_generation_prompt, model, parser)

summaize_report_chain = RunnableSequence(summarize_report_prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) >5000000, summaize_report_chain),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

response = final_chain.invoke({"topic": "Russia vs Ukraine"})
print(response)
