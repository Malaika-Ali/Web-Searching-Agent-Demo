import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool 

from langchainhub import Client

from exa_py import Exa
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent
from langchain.tools import tool
load_dotenv()
EXA_KEY=os.getenv("EXA_API_KEY")
exa = Exa(api_key=EXA_KEY)
# results = exa.search(
#     "entry level jobs for software engineers",
#     num_results=2
# )

# print(results)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_retries=5,
)

# langchain hub gives tons of predefined prompts
client = Client()
prompt = client.pull("hwchase17/react")

# tools=[results]


exa = Exa(api_key=os.getenv("EXA_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""
    response = exa.search(
        query,
        num_results=5
    )

    return str(response)

tools = [search_web]

agent=create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt
)

response = agent.invoke({
    "messages": [ 
        {
            "role": "user",
            "content": "Find the capital of Pakistan."
        }
    ]
})

print(response)