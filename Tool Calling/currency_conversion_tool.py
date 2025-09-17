from langchain_core.tools import InjectedToolArg, tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from typing import Annotated
from dotenv import load_dotenv
import os
import requests

load_dotenv()


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """This function fetches the exchange rate from one currency to another."""
    url = f"https://v6.exchangerate-api.com/v6/{os.getenv('EXCHANGE_API')}/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    data = response.json()
    return data


@tool
def currency_convertor(
    base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Convert amount from one currency to another using a fixed exchange rate."""
    return base_currency_value * conversion_rate


llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
    temperature=0,
)

llm_with_tools = llm.bind_tools([get_exchange_rate, currency_convertor])

chat = [
    HumanMessage(
        content="What is the exchange rate between USD and INR? Based on that convert 100 USD to INR."
    ),
]

# tool_reponse = llm_with_tools.invoke(chat)
# chat.append(tool_reponse)
# print(tool_reponse.tool_calls)

# for tool_call in tool_reponse.tool_calls:
#     # Execute the first tool call
#     if tool_call['name'] == "get_exchange_rate":
#         tool_message = get_exchange_rate.invoke(tool_call)
#         chat.append(tool_message)
#         conversion_rate = json.loads(tool_message.content)['conversion_rate']
#         # print(f"Conversion Rate: {conversion_rate}")
        
#     if tool_call['name'] == "currency_convertor":
#         tool_call['args']['conversion_rate'] = conversion_rate
#         tool_message = currency_convertor.invoke(tool_call)
#         chat.append(tool_message)
    
# # chat.append(final_response)
# res = llm_with_tools.invoke(chat)
# print(res)  # should print the converted amount
# print()
# print(chat)


## Agentic Flow

agent_executor = initialize_agent(
    tools = [get_exchange_rate, currency_convertor],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query = "What is the exchange rate between INR and AUD? What will be 30000000 INR in AUD?"
response = agent_executor.invoke(query)

print(response)  # should print the converted amount

