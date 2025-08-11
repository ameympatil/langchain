from langchain import vectorstores
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0,
    )
    return model


model = get_gemini_model()

loader = TextLoader(r"D:\Study\langchain\data\data.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=216, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

vectorstores = FAISS.from_documents(
    texts, GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

retriever = vectorstores.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=model, chain_type="stuff", retriever=retriever
)

query = input("Enter a query: ")
response = qa_chain.invoke({"query": query})

print(response)
