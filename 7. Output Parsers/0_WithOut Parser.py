from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# prompts
from langchain_core.prompts import PromptTemplate

load_dotenv()
'''
llm = HuggingFaceEndpoint( 
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
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


# Old method :-
prompt1 = templet_1.invoke({'topic': "Black-Hole"})
result = model.invoke(prompt1)

prompt2 = templet_2.invoke( { "text": result.content} )
result = model.invoke(prompt2)

print(result.content)