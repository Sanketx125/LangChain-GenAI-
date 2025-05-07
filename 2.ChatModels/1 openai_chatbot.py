from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatbot = ChatOpenAI(model= 'gpt-4',
                     temperature= 0.3,   # creativity { [0.0-0.3 -->math, code ] ,[0.4-0.7---> normal test], [1.5+ --> more creative, poem, songs] }
                     max_completion_tokens= 10,  # restrict number of words in respnces


                    )

responce = chatbot.invoke("what is the addition of 3 and 2 ?")

print(responce.content)