from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
import json


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.5, api_key=os.getenv("GROQ_API_KEY")
)

schema = [
    ResponseSchema(name="fact1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact3", description="Fact 3 about the topic"),
]

json_parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write me 3 facts about the topic: {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": json_parser.get_format_instructions()},
)

chain = template | model | json_parser

result = chain.invoke({"topic":"Data Science"})

print(json.dumps(result)) 
