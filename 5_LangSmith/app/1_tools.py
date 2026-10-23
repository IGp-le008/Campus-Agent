# -----------------------------------------Dependencies-----------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage


# -----------------------------------------Load Environment Variables-----------------------------------------

# Load environment variables from parent directory
ENV_PATH = Path("/home/program/Documents/Tutor/Campus/5_LangSmith/.env")
load_dotenv(ENV_PATH)

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key:
    print(f"✓ API Key loaded: {api_key[:20]}...")
else:
    print("✗ API Key not found")


# -----------------------------------------Initialize LLM-----------------------------------------

# Initialize the LLM with OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="nvidia/nemotron-3.5-lightning:free",
)

print("✓ LLM initialized with OpenRouter")
print(f"  Model: nvidia/nemotron-3.5-lightning:free")


# -----------------------------------------Define Chat State-----------------------------------------

# Define the state structure for the chatbot
class ChatState(TypedDict):
    """State structure for the chat graph"""
    message: Annotated[list[BaseMessage], add_messages]


# -----------------------------------------Define Chat Node-----------------------------------------

def ChatNode(state: ChatState):
    """
    Chat node that processes user messages and returns LLM response.

    Args:
        state: ChatState containing list of messages

    Returns:
        dict with updated message list including LLM response
    """
    user_message = state['message']
    llm_response = llm.invoke(user_message)
    return {'message': [llm_response]}


# -----------------------------------------Build the Graph-----------------------------------------

# Create the state graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node('ChatNode', ChatNode)

# Add edges: START -> ChatNode -> END
graph.add_edge(START, 'ChatNode')
graph.add_edge('ChatNode', END)

print("✓ Graph structure created: START -> ChatNode -> END")


# -----------------------------------------Compile the Graph with Memory Checkpointer-----------------------------------------

# Create a memory checkpointer for conversation history
checkpointer = InMemorySaver()

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

print("✓ Graph compiled with InMemorySaver checkpointer")


# -----------------------------------------Test the Chatbot-----------------------------------------

def test_chatbot():
    """Test the chatbot with a sample message"""
    user_message = input("Enter here :")
    thread_id = "test-thread-1"

    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }

    print(f"\n📝 User Message: {user_message}")
    print("⏳ Waiting for response...\n")

    # Stream the response
    response = chatbot.invoke({'message':[HumanMessage(content=user_message)]},config = config)
    print(f"Assistant response: {response['message'][-1].content}")

    return response


# -----------------------------------------Graph Visualization-----------------------------------------

def visualize_graph():
    """
    Visualize the graph structure

    Returns the graph image or prints ASCII representation
    """
    try:
        from IPython.display import Image, display
        # Generate graph image
        graph_image = chatbot.get_graph().draw_mermaid_png()
        display(Image(graph_image))
        print("✓ Graph visualization displayed")
    except Exception as e:
        print(f"Note: Visualization requires graphviz. Error: {e}")
        print("\nGraph Structure:")
        print("  START")
        print("    ↓")
        print("  ChatNode (processes messages with LLM)")
        print("    ↓")
        print("  END")


# -----------------------------------------Main Execution-----------------------------------------

if __name__ == "__main__":
    # Test the chatbot


    test_chatbot()


    # Visualize the graph
    visualize_graph()
