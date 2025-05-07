# generate embedding of multiple queries


from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions= 32)

documents= [

    "hello what is your name ?",
    "do you like danceing ?",
    "you are so smart"
]

responce = embedding.embed_documents(documents)

# converting embedding responce into string
responce = str(responce)

print(responce)