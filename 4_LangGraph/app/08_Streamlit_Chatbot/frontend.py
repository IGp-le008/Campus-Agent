import streamlit as st
from backend import chat_bot
from langchain.messages import HumanMessage

THREAD_ID = '1'

CONFIG = {
    'configurable' : {
        'thread_id' : THREAD_ID
    }
}

#To iniitalize the session state to store message history
if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]

st.title("LangGraph ChatBot")

#Dispaly all the message history
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['message'])


user_msg = st.chat_input("Type here!")

if user_msg:

    #Append user message to session state
    st.session_state['msg_history'].append({
        'role' : 'user',
        'message' : user_msg
    })

    with st.chat_message('user'):
        st.text(user_msg)

    #Append assistant message to session state
    # Invoke the LangGraph chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_msg = st.write_stream(
               ( 
                message_chunk.content for message_chunk, metadata in chat_bot.stream(
                    {
                    "messages": [
                        HumanMessage(content=user_msg)
                    ]
                },
                config=CONFIG,
                stream_mode = 'messages'
                )
                if message_chunk.content
                )
            )
            
        # Store the assistant response for subsequent reruns
        st.session_state["msg_history"].append({
            "role": "assistant",
            "message": ai_msg
        })

