from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# model
chatbot = ChatGoogleGenerativeAI( model='gemini-1.5-pro')

# website header
st.header("Research Tool")

# inputbox in webiste
query = st.text_input("what is your name ?")

# button
if st.button("Summerize"):
    responce = chatbot.invoke(query)
    st.text(responce.content)
