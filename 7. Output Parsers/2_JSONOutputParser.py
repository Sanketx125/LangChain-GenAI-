from langchain_google_genai import ChatGoogleGenerativeAI
# load API
from dotenv import load_dotenv
# prompts
from langchain_core.prompts import PromptTemplate
# Str Output Parser
from langchain_core.output_parsers import JsonOutputParser

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

# parser
parser = JsonOutputParser()

# templet
templet = PromptTemplate(
    template="Give me the name, age and city of a fictional Perosn \n {format_instruction}",
    input_variables= [],
    partial_variables= {"format_instruction": parser.get_format_instructions()}
)


# Chain
Chain = templet | model | parser

# result
final_result = Chain.invoke({})
print(final_result)
print(type(final_result))
