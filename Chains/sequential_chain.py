from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()

report_prompt = PromptTemplate(
    template="Generate a detailed report on : {topic}",
    input_variables=["topic"],
)

summary_prompt = PromptTemplate(
    template="Generate a 5 line summary for the report: \n {report}",
    input_variables=["report"],
)

model = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

parser = StrOutputParser()

chain = report_prompt | model | parser | summary_prompt | model | parser

result = chain.invoke({"topic": "Un employment in india"})
print(result)

chain.get_graph().print_ascii()
