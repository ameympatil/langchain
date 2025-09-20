from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, tool
from dotenv import load_dotenv
import os
from langchain import hub

load_dotenv()

llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
    temperature=0,
)

# Define the tool
search = DuckDuckGoSearchRun()


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Dummy implementation, replace with actual weather fetching logic
    return f"The current weather in {location} is sunny with a temperature of 25°C."


# Load the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=[search, get_weather],
    prompt=prompt,
)

# wrap it in an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search, get_weather],
    verbose=True,
)

# Invoke the agent
query = "What is the weather in Goldcoast Australia?"

response = agent_executor.invoke({"input": query})
print(response)
