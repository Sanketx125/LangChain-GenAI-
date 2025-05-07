from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# model
chatbot = ChatGoogleGenerativeAI( model='gemini-1.5-pro')

# message
message = [

    SystemMessage(content= "You are a most advanced personal AI assistant"),
    HumanMessage(content= "tell me about langchain")
]

responce = chatbot.invoke(message)

message.append(AIMessage(responce.content))

print(message)