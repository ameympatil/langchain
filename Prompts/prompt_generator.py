from langchain_core.prompts import PromptTemplate

## Template
template = PromptTemplate(
    template="""
    You are a research assistant.
    You are given a research paper: {selected_paper}, a style: {style_input}, and a length: {length_input}.
    You need to generate a summary of the paper in the given style and length.
    """,
    input_variables=["selected_paper", "style_input", "length_input"]
)

template.save("template.json")