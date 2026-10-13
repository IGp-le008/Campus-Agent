import streamlit as st

# Box area for user Chat 

with st.chat_message('user'):
    st.text('Hi!')

# Box area for Ai Chat

with st.chat_message('assistant'):
    st.text("Hello how can I help you?")

user_input = st.chat_input("Type here")


if user_input:
    with st.chat_message('user'):
        st.text(user_input)