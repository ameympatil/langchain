from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model1 = ChatGroq(model="llama-3.3-70b-versatile")
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

prompt1 = PromptTemplate(
    template="Generate short and simple notes for this topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Create a 5 questions and ansers Quiz on this topic: {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \nNotes: {notes} \nQuiz: {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

response = chain.invoke({"topic": "Machine Learning"})
print(response)


chain.get_graph().print_ascii()
