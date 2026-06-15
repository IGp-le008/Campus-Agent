import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal, Optional

load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

class Review(BaseModel):
    key_themes:list[str]=Field(description="Write a short summarized key theme of the discussed review")
    summary:str=Field(description="Provide a brief summary of the review in about 30-80 words")
    sentiment:Literal["pos","neg"]=Field(description="Return the overall sentiment of the review as pos for positive and neg as negative")
    pros:Optional[list[str]]=Field(description="Write down all the pros")
    cons:Optional[list[str]]=Field(description="Write down all the cons")


structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""

I recently upgraded to the Quantum Nexus 10, and it has easily exceeded my expectations for a daily driver. The standout feature is definitely the stunning 120Hz AMOLED display, which makes scrolling through social media and watching videos incredibly smooth and vibrant. Battery life is also a major pro, consistently pushing past a day and a half of heavy use, paired with blazing-fast 65W charging that saves me in a pinch. On the downside, the phone is quite a fingerprint magnet due to its glossy glass back, and the ultrawide camera struggles a bit in low-light conditions, producing slightly grainy shots. There is also a bit of pre-installed bloatware that takes a few minutes to manually clean up. Despite these minor quirks, the blazing performance and gorgeous screen make it an absolute joy to use, offering incredible premium value that I would highly recommend to anyone.

""")



print("-----------------------------------------Structured Output Using Pydantic-------------------------------------------\n\n")


print(result)



print("----------------------------------------------------------------------------------------------------------------------")

