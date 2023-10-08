import streamlit as st
from chatbot import chat_with_bot  # Import the chatbot function from your chatbot code

st.title("AI Chatbot")

# Initial greeting
st.write("AI: Hello! How can I assist you today?")

# User input textbox
user_input = st.text_input("User:")

if st.button("Submit"):
    if user_input:
        # Get the chatbot response
        response = chat_with_bot(user_input)
        st.write("AI:", response)

# Exit button
if st.button("Exit"):
    st.write("AI: Goodbye! Have a great day!")

