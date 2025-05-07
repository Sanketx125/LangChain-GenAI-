from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# model
model = ChatGoogleGenerativeAI( model='gemini-1.5-pro')

# chat history
chat_history = [
                SystemMessage(content= "You are a most advanced personal AI assistant & your name is Eva"),
                ]


# conversation
print("---------------------------------------\n")
while True:

    your_input = input("\nYou: ")
    your_input = HumanMessage(content = your_input)
    chat_history.append(your_input)

    if your_input in ['exit', 'bye']:
        break
    responce = model.invoke(chat_history)
    chat_history.append(AIMessage(content= responce.content))
    print("\nAI: ", responce.content)


print(chat_history)