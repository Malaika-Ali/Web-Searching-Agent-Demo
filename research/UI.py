import os
import streamlit as st
from dotenv import load_dotenv

from exa_py import Exa
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

EXA_KEY = os.getenv("EXA_API_KEY")

if not EXA_KEY:
    st.error("EXA_API_KEY was not found in your .env file.")
    st.stop()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="centered"
)


# --------------------------------------------------
# Exa setup
# --------------------------------------------------

exa = Exa(api_key=EXA_KEY)


# --------------------------------------------------
# Gemini model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_retries=5,
)


# --------------------------------------------------
# Web search tool
# --------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""

    response = exa.search(
        query,
        num_results=5
    )

    return str(response)


tools = [search_web]


# --------------------------------------------------
# Create agent
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are a helpful AI research assistant.

    You can search the web using the search_web tool.

    Use the search_web tool whenever the user asks for:
    - current information
    - recent events
    - news
    - facts that may have changed
    - information you are unsure about

    Give clear, concise and useful answers.
    """
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🔎 AI Research Agent")

st.caption(
    "Ask a question and the AI agent can search the web using Exa."
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Researching..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    }
                )

                # Get final assistant message
                assistant_message = response["messages"][-1].content

                st.markdown(assistant_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )