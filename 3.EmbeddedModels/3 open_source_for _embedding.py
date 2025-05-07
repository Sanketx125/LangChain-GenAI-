from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "do you like singing ?"

# here we can get vector embed
responce = embedding.embed_query(text)

print(str(responce))