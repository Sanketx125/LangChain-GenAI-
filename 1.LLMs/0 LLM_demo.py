from langchain_openai import OpenAI
from dotenv import load_dotenv   # accessing data from .evn file

load_dotenv()  

llm = OpenAI( model='gpt-3.5-turbo-instruct')

result = llm.invoke('hie')

print(result)
