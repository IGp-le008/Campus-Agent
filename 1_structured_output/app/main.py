import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Optional,Literal, Annotated

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the reivew in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal["pros","neg"],"Return sentiment of the review either negative, positive or neutral"]
    pros:Annotated[Optional[list[str]],"write down all the pros inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside a list"]

structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""


I recently upgraded to the Quantum Nexus 10, and it has easily exceeded my expectations for a daily driver. The standout feature is definitely the stunning 120Hz AMOLED display, which makes scrolling through social media and watching videos incredibly smooth and vibrant. Battery life is also a major pro, consistently pushing past a day and a half of heavy use, paired with blazing-fast 65W charging that saves me in a pinch. On the downside, the phone is quite a fingerprint magnet due to its glossy glass back, and the ultrawide camera struggles a bit in low-light conditions, producing slightly grainy shots. There is also a bit of pre-installed bloatware that takes a few minutes to manually clean up. Despite these minor quirks, the blazing performance and gorgeous screen make it an absolute joy to use, offering incredible premium value that I would highly recommend to anyone.



""")


print(result)

