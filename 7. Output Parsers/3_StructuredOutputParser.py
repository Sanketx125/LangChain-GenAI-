from langchain_google_genai import ChatGoogleGenerativeAI
# load API
from dotenv import load_dotenv
# prompts
from langchain_core.prompts import PromptTemplate
# Parser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')


# structure (schema)
schema = [
    ResponseSchema(name= 'Fact 1', description="Fact 1 about the Topic"),
    ResponseSchema(name= 'Fact 2', description="Fact 2 about the Topic"),
    ResponseSchema(name= 'Fact 3', description="Fact 3 about the Topic"),
    ResponseSchema(name= 'Fact 4', description="Fact 4 about the Topic"),

]

# Parser
parser = StructuredOutputParser.from_response_schemas(schema)

# templete
templete = PromptTemplate(
    template="Give 11 facts about {topic} \n {format_instruction}",
    input_variables= ["topic"],
    partial_variables= {"format_instruction": parser.get_format_instructions()}
)


Chain = templete | model | parser


result  = Chain.invoke({'topic': 'Human vagina'})

print(result)