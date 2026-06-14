from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypeDict

load_dotenv()
model=ChatOpenAI()

class Review(TypeDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the reivew in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal["pros","neg"],"Return sentiment of the review either negative, positive or neutral"]
    pros:Annotated[Optional[list[str]],"write down all the pros inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside a list"]

structured_model=model.with_structured_output(Review)
result=structuredmodel.invoke("""Today is Sunday and it's sunny. I am very happy""")
print(result)
