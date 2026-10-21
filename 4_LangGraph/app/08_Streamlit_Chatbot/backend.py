
# -----------------------------------------Dependencies-----------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import json
from datetime import datetime


from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage


# -----------------------------------------ENV & LLM-----------------------------------------

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = None

# Initialize the model
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="nvidia/nemotron-3.5-lightning:free",
)


# ----------------------------------------- Chat State -----------------------------------------


class ChatState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]


def ChatNode(state:ChatState):
    user_message = state['message']
    llm_response = llm.invoke(user_message)
    return {'message' : [llm_response]}


# ----------------------------------------- Graph and Nodes -----------------------------------------

graph = StateGraph(ChatState)
graph.add_node('ChatNode', ChatNode)
graph.add_edge(START, 'ChatNode')
graph.add_edge('ChatNode', END)

# Creating checkpoint
checkpoint = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpoint)


# ----------------------------------------- RAM-Based Chat History Storage -----------------------------------------
# This uses in-memory storage so it works seamlessly when hosted on GitHub or any cloud platform
# Data persists during the app session but resets on app restart
# Perfect for GitHub hosting where you can't write files to the server

CHAT_HISTORY_STORE = {}


def get_chat_preview(messages, max_chars=50):
    """Get a preview of the chat for display in history."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            preview = msg.get('message', '')[:max_chars]
            return preview + '...' if len(msg.get('message', '')) > max_chars else preview
    return "New Chat"


def save_chat_session(thread_id, messages):
    """Save a chat session to RAM (in-memory storage)."""
    thread_id_str = str(thread_id)

    CHAT_HISTORY_STORE[thread_id_str] = {
        'thread_id': thread_id_str,
        'messages': messages,
        'timestamp': datetime.now().isoformat(),
        'preview': get_chat_preview(messages)
    }

    return CHAT_HISTORY_STORE


def load_chat_session(thread_id):
    """Load a specific chat session from RAM."""
    return CHAT_HISTORY_STORE.get(str(thread_id), {}).get('messages', [])


def get_all_chats():
    """Get all chats sorted by timestamp (latest first)."""
    chats = list(CHAT_HISTORY_STORE.values())
    # Sort by timestamp, latest first
    chats.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return chats