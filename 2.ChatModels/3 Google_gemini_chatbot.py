from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatbot = ChatGoogleGenerativeAI( model='gemini-1.5-pro')

responce = chatbot.invoke('what is your name ?')

print(responce.content)