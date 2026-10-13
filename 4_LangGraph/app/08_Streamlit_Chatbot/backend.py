import streamlit as st

#To iniitalize the session state to store message history
if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]


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
    st.session_state['msg_history'].append({
        'role' : 'assistant',
        'message' : user_msg
    })

    with st.chat_message('assistant'):
        st.text(user_msg)

    
