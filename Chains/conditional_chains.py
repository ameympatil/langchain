from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model1 = ChatGroq(model="llama-3.3-70b-versatile")
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template="Wrute an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Wrute an appropriate response to this Negative feedback: \n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model1 | parser),
    (lambda x: x.sentiment == "negative", prompt2 | model1 | parser),
    RunnableLambda(lambda x: "couldn't find sentiment"),
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a best  phone"}))
