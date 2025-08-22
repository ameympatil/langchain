from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="")

loader = TextLoader("demo.txt")

docs = loader.load()

pages = splitter.split_documents(docs)

text = """Sachin Tendulkar is a former Indian cricketer widely regarded as one of the greatest batsmen in the history of cricket. Born on April 24, 1973, in Mumbai, India, Tendulkar made his debut for the Indian national team at the age of 16. Over a career spanning 24 years, he set numerous records, including being the highest run-scorer in both Test and One Day International (ODI) cricket. He was the first player to score 100 international centuries and the first to reach 200 runs in an ODI match. Tendulkar's skill, dedication, and sportsmanship have made him a beloved figure in India and around the world. He retired from international cricket in 2013 and was awarded the Bharat Ratna, India's highest civilian honor, for his contributions to the sport."""


docs = splitter.split_text(text)

print(pages)
