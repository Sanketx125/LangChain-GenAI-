from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

# Load API 
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature= 0.7)

# Define schema for structured output
class Review(TypedDict):

    key_themes : Annotated[list[str], "write down all the key themes discussed in the review in list"]
    summary: Annotated[str, "A brief summery of the review"] 
    sentiment: Annotated[str, "Return sentiment of the review either Negative, Positive or Neutral"]
    #pros: Annotated[Optional[list[str]], "write down the all pros inside the list"]
    #cons: Annotated[Optional[list[str]], "write down the all cons inside the list"]
     


# Structured output model
struc_model = model.with_structured_output(Review)

# Generate structured response
response = struc_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""
)

print(response)
print("""
      --->
      ------>
      ---------->""")

#print(response['sentiment'])
print("""
      --->
      ------>
      ---------->""")