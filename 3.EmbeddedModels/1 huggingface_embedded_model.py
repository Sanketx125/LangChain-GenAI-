from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions= 32)

responce = embedding.embed_query("what is your name ?")

# converting embedding responce into string
responce = str(responce)

print(responce)