from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY")
)

topic = "black hole"

# 1: detailed report
detailed_prompt_template = PromptTemplate(
    template="Write a detailed report for the topic: {topic}"
)

# 2: Summary(5 lines)
summary_prompt_template = PromptTemplate(
    template="Write a 5 line summary for the report: {report}"
)

## Output Parser
parser = StrOutputParser()

chain = (
    detailed_prompt_template | model | parser | summary_prompt_template | model | parser
)

print(chain.invoke({"topic": topic}))


# detailed_prompt = detailed_prompt_template.invoke({"topic": topic})
# report = model.invoke(detailed_prompt).content

# summary_prompt = summary_prompt_template.invoke({"report": report})
# summary = model.invoke(summary_prompt).content

# print("Report \n", report)
# print("\n\nSummary \n", summary)
