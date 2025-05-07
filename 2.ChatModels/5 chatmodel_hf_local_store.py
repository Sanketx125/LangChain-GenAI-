from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id= 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs= dict(
        temperature = 0.5,
        max_new_tokens = 100,
        do_sample=True
    )
)

model = ChatHuggingFace(llm= llm)

responce = model.invoke("what is the addition of 3 and 2 ?")

print(responce)