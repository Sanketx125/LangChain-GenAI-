from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

# model
chatbot = ChatGoogleGenerativeAI( model='gemini-1.5-pro')

# website header
st.header("Research Tool")


paper_input = st.text_input( "Enter Research Paper Name")

style_input = st.selectbox( "Select Explanation Style", ["Beginner", "Advanced", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["very small", "Short", "Medium", "Long"] )

# template
template = load_prompt('template.json')

# fill placeholder
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input' : style_input,
    'length_input': length_input
    })

# button
if st.button("Summerize"):
    responce = chatbot.invoke(prompt)
    st.write(responce.content)
