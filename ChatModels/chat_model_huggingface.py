from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFacePipeline(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0, "max_new_tokens": 100},

)

model = ChatHuggingFace(
    llm=llm,
)

response = model.invoke("What is the capital of India?")
print(response)
