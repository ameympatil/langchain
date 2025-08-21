from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-001",
        temperature=0,
    )
    return model


model = get_gemini_model()

template = PromptTemplate(
    template="Answer the user query based on the content below: \n {content}\n\nquery:{query}",
    input_variables=["content", "query"],
)

loader = WebBaseLoader(
    web_path="https://www.amazon.in/iPhone-16-128-GB-Control/dp/B0DGJH8RYG/ref=sr_1_4?adgrpid=172137495971&hvadid=714735660631&hvdev=c&hvlocphy=9300430&hvnetw=g&hvqmt=e&hvrand=17660445571579158611&hvtargid=kwd-2266317994499&hydadcr=25227_2856623&mcid=15e889146bc7307eb70a1b71b21dbeac&sr=8-4"
)
docs = loader.load()

content = docs[0].page_content

parser = StrOutputParser()

chain = template | model | parser

query = "What is the price of iphone 16 now? And tell me the color and specifications of the phone?"

print(chain.invoke({"query":query, "content":docs[0].page_content}))
