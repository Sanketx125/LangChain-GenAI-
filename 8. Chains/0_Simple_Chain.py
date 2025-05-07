from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
topic = input("Enter a Topic : ")

tamplate = PromptTemplate(
    template="Generate 5 intersting facts on the {topic}",
    input_variables= ['topic']
)

model = ChatGoogleGenerativeAI(model= 'gemini-1.5-pro')

parser = StrOutputParser()


Chain = tamplate | model | parser


result = Chain.invoke({'topic': topic})

print(result)

Chain.get_graph().print_ascii()
