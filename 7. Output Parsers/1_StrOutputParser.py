from langchain_google_genai import ChatGoogleGenerativeAI
# load API
from dotenv import load_dotenv
# prompts
from langchain_core.prompts import PromptTemplate
# Str Output Parser
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

'''
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

llm = HuggingFaceEndpoint( 
    repo_id="google/gemma-3-1b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

'''
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# 1st Prompt  ---> Detailed report
templet_1 = PromptTemplate(

    template="Write a detailed report on {topic}",
    input_variables= ['topic']
)

# 2nd Prompt   ---> Summery of Report
templet_2 = PromptTemplate(

    template="Write a 5 line summery on following text. /n {text}",
    input_variables= ['text']
)


# creating parser:  StrOuputParser are always use with chains
parser = StrOutputParser()

# Chains
Chain = templet_1 | model | parser | templet_2 | model | parser  # >>> this is how chains are look like in langchain

result = Chain.invoke({'topic': 'black hole'})

print(result)