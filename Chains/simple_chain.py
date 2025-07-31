from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about the topic: {topic}",
    input_variables=["topic"],
)


model = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Badminton"})
print(result)


chain.get_graph().print_ascii()
