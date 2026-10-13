
import os
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# Load the .env file from the Campus directory
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(ENV_PATH)

# Validate the API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

# Initialize the model
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="nvidia/nemotron-3.5-lightning:free",
)


# Define the graph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Create the chatbot node
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Create the graph
builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# Enable conversation memory
checkpointer = InMemorySaver()

chat_bot = builder.compile(checkpointer=checkpointer)


# Configure the conversation thread
# config = {
#     "configurable": {
#         "thread_id": "thread-0"
#     }
# }


# Run the chatbot interactively
# if __name__ == "__main__":
#     while True:
#         user_input = input("You: ")

#         if user_input.strip().lower() in {"exit", "quit"}:
#             break

#         result = chat_bot.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config=config,
#         )

#         print("Bot:", result["messages"][-1].content)

