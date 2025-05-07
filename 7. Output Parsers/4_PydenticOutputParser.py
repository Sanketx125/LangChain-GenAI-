from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# pydentic object act like schema
class Person (BaseModel):

    name: str = Field( description= "Name of the Person")
    age: int = Field( gt = 18, description= "age of the Person")
    city: str = Field( description= "Name of the city Person belongs to")

# Parser
parser = PydanticOutputParser(pydantic_object= Person)

# Chain
template = PromptTemplate(
    template="give me name, age & city of fictional {place} person. \n {format_instruction}",
    input_variables=['place'],
    partial_variables=  { "format_instruction": parser.get_format_instructions()}
)

#Chain
Chain = template | model | parser

result = Chain.invoke({'place': 'india'})


print(result)
