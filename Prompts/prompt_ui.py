from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash-001",
)


st.header("Research Tool")

selected_paper = st.selectbox("Select a research paper", options=["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select a style", options=["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Enter the length of the summary",["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

include_citations = st.checkbox("Include citations")

print(include_citations)

template = load_prompt("template.json")

if st.button("Generate"):
    chain = template | model
    st.text("Generating...")
    response = chain.invoke({
    "selected_paper": selected_paper, 
    "style_input": style_input,  
    "length_input": length_input
    })
    st.write(response.content)
