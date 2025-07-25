from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# model = ChatGroq(
#     model="qwen/qwen3-32b",
#     temperature=0,
#     api_key=os.getenv("GROQ_API_KEY"),
# )


def get_gemini_model():
    model = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-lite",
        temperature=0,
    )
    return model


model = get_gemini_model()


# schema
class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review in one word: either negative, positive or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list only if they are explicitly present in the review",
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list only if they are explicitly present in the review",
    )
    name: Optional[str] = Field(
        default=None, description="Write the name of the reviewer"
    )


structured_model = model.with_structured_output(schema=Review)


result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
                                 
""")

print(result.model_dump_json())
# print(result)
