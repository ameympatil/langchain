from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


code = """
class TextSplitterDemo:
    def split_sample_text(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        sample_text = "LangChain is a framework for developing applications powered by language models. It enables easy text splitting and processing."
        chunks = text_splitter.split_text(sample_text)
        print(chunks)

# Example usage
if __name__ == "__main__":
    demo = TextSplitterDemo()
    demo.split_sample_text()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=1000, language=Language.PYTHON, chunk_overlap=200
)

docs = splitter.split_text(code)

print(docs)
