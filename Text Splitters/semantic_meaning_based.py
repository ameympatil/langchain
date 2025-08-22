## Example of Semantic Meaning Based Text Splitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv

load_dotenv()


loader = TextLoader("semantic_text.txt")

docs = loader.load()
splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    breakpoint_threshold_amount=1,
    breakpoint_threshold_type="standard_deviation"
)

chunks = splitter.split_documents(docs)

print(chunks)
