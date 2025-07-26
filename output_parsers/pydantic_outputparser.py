from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=1, api_key=os.getenv("GROQ_API_KEY")
)


# schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the preson belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

prompt_template = PromptTemplate(
    template="Generate the name, age, and city of a Fictional {place} person \n  {format_instruction}",
    input_variables=["person"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = prompt_template.invoke({"place": "Australia"})

# print(prompt)

# result = model.invoke(prompt).content

# print(parser.parse(result))

chain = prompt_template | model | parser

result = chain.invoke({"place": "Indian"})
print(result)
