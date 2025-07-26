from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.5, api_key=os.getenv("GROQ_API_KEY")
)

json_parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": json_parser.get_format_instructions()},
)

# prompt = template.invoke({})

# response = model.invoke(prompt)

# final_result = json_parser.parse(response.content)

# print(final_result)

chain = template | model | json_parser

result = chain.invoke({})

print(result)