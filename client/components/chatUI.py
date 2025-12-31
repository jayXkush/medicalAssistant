import streamlit as st
from utils.api import ask_question


def render_chat():
    st.subheader("💬 Chat with your assistant")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    # render existing chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # input and response
    user_input=st.chat_input("Type your question....")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role":"user","content":user_input})

        with st.spinner("Thinking..."):
            response=ask_question(user_input)
            if response.status_code==200:
                data=response.json()
                answer=data.get("response", "I'm sorry, I couldn't generate a response.")
                sources=data.get("sources",[])
                st.chat_message("assistant").markdown(answer)
                # if sources:
                #     st.markdown("📄 **Sources: **")
                #     for src in sources:
                #         st.markdown(f"- `{src}`")
                st.session_state.messages.append({"role":"assistant","content":answer})
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text
                st.error(f"Error: {error_msg}")
                st.session_state.messages.append({"role":"assistant","content":f"Error: {error_msg}"})
