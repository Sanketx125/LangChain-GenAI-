from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatbot = ChatAnthropic(model = 'claude-3-7-sonnet-20250219',
                        temperature= 0.3,
                        max_tokens_to_sample= 10 )

responce = chatbot.invoke('kaisa he re ba ba.')

print(responce.content)