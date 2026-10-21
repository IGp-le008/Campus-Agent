
# -----------------------------------------Dependencies-----------------------------------------

import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from backend import chatbot, save_chat_session, load_chat_session, get_all_chats


# -----------------------------------------Page Configuration-----------------------------------------

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------Custom CSS Styling-----------------------------------------

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* Sidebar header */
    .sidebar-header {
        font-size: 24px;
        font-weight: 700;
        color: #000;
        margin-bottom: 24px;
        padding: 16px 0;
    }

    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 100%;
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
    }

    /* Message styling */
    .user-message {
        background: #10a37f;
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        word-wrap: break-word;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .assistant-message {
        background: #f0f0f0;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        word-wrap: break-word;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Logo and header styling */
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        text-align: center;
    }

    .logo-text {
        font-size: 48px;
        font-weight: 700;
        color: #10a37f;
        margin: 16px 0;
        letter-spacing: -1px;
    }

    .logo-subtext {
        font-size: 16px;
        color: #8b8b8b;
        margin-top: 8px;
    }

    /* Input area styling */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 260px;
        right: 0;
        background: #ffffff;
        border-top: 1px solid #e0e0e0;
        padding: 16px 40px;
        z-index: 100;
    }

    /* Chat history button styling */
    .history-button {
        width: 100%;
        text-align: left;
        padding: 12px 16px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background: #f9f9f9;
        color: #333;
        margin-bottom: 8px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
    }

    .history-button:hover {
        background: #ececf1;
        border-color: #d0d0d0;
    }

    /* New Chat button styling */
    .new-chat-button {
        width: 100%;
        padding: 12px 16px;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        background: #ffffff;
        color: #333;
        font-weight: 600;
        margin-bottom: 20px;
        transition: all 0.2s;
    }

    .new-chat-button:hover {
        background: #f0f0f0;
        border-color: #10a37f;
        color: #10a37f;
    }

    /* Chat history section */
    .history-section {
        margin-top: 24px;
    }

    .history-title {
        font-size: 12px;
        font-weight: 600;
        color: #8b8b8b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding: 0 8px;
    }

    /* Separator line */
    .separator {
        height: 1px;
        background: #e0e0e0;
        margin: 16px 0;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #888;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------Session State Initialization-----------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if 'sidebar_open' not in st.session_state:
    st.session_state['sidebar_open'] = True

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []


# -----------------------------------------Helper Functions-----------------------------------------

def generate_thread_id():
    """Generate a new thread ID."""
    return str(uuid.uuid4())


def reset_chat():
    """Reset chat to start a new conversation."""
    new_thread_id = generate_thread_id()
    st.session_state['thread_id'] = new_thread_id
    st.session_state['message_history'] = []
    st.rerun()


def load_chat_by_thread(thread_id):
    """Load a previous chat session by thread ID."""
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = load_chat_session(thread_id)
    st.rerun()


def format_timestamp(iso_string):
    """Format ISO timestamp to readable format."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso_string)
        # Show time if today, else show date
        from datetime import date
        today = date.today()
        msg_date = dt.date()

        if msg_date == today:
            return dt.strftime("%I:%M %p")
        elif msg_date.year == today.year:
            return dt.strftime("%b %d")
        else:
            return dt.strftime("%b %d, %Y")
    except:
        return iso_string


# -----------------------------------------Sidebar Layout-----------------------------------------

with st.sidebar:
    # Sidebar header with toggle
    col1, col2 = st.columns([1, 0.15])

    with col1:
        st.markdown('<div class="sidebar-header">💬 My Chats</div>', unsafe_allow_html=True)

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        reset_chat()

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # Chat History Section
    st.markdown('<div class="history-title">History</div>', unsafe_allow_html=True)

    # Load and display all chats
    all_chats = get_all_chats()

    if all_chats:
        for chat in all_chats:
            thread_id = chat.get('thread_id')
            preview = chat.get('preview', 'Chat')

            # Highlight current chat
            is_current = thread_id == st.session_state['thread_id']
            button_style = "background: #ececf1; border-color: #10a37f;" if is_current else ""

            if st.button(
                f"💬 {preview}",
                key=f"chat_{thread_id}",
                use_container_width=True,
                help=f"Click to open this chat"
            ):
                load_chat_by_thread(thread_id)
    else:
        st.markdown(
            '<div style="text-align: center; color: #999; padding: 20px 0; font-size: 14px;">No chat history yet</div>',
            unsafe_allow_html=True
        )


# -----------------------------------------Main Chat Area-----------------------------------------

# Logo and greeting for empty chat
if not st.session_state['message_history']:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
            <div class="logo-container">
                <div style="font-size: 64px; margin-bottom: 16px;">🤖</div>
                <div class="logo-text">LangGraph</div>
                <div class="logo-subtext">AI-Powered Chatbot</div>
            </div>
        """, unsafe_allow_html=True)
else:
    # Display chat history
    for message in st.session_state['message_history']:
        role = message.get('role', 'user')
        content = message.get('message', '')

        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(content)


# -----------------------------------------Chat Input & Processing-----------------------------------------

# Fixed input area at bottom
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({
        'role': 'user',
        'message': user_input
    })

    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Get AI response with streaming
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                config = {
                    'configurable': {
                        'thread_id': st.session_state['thread_id']
                    }
                }

                # Stream response from LLM
                llm_response = st.write_stream(
                    (
                        message_chunk.content for message_chunk, metadata in chatbot.stream(
                            {
                                "message": [
                                    HumanMessage(content=user_input)
                                ]
                            },
                            config=config,
                            stream_mode='messages'
                        )
                        if message_chunk.content
                    )
                )

                # Store assistant response in history
                st.session_state['message_history'].append({
                    'role': 'assistant',
                    'message': llm_response
                })

                # Save chat session to persistent storage
                save_chat_session(st.session_state['thread_id'], st.session_state['message_history'])

                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please make sure your OPENROUTER_API_KEY is set correctly in the .env file")
