from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
topic = input("Enter a Topic : ")

tamplate1 = PromptTemplate(
    template="create deailed report on {topic}",
    input_variables= ['topic']
)

tamplate2 = PromptTemplate(
    template="give most 5 importent point on the {text}",
    input_variables= ['text']
)


model = ChatGoogleGenerativeAI(model= 'gemini-1.5-pro')

parser = StrOutputParser()


Chain = tamplate1 | model | parser | tamplate2 | model | parser


result = Chain.invoke({'topic': topic})

print(result)


