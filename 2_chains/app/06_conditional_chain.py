import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)   


class FeedBack(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description="Return the sentiment of the feedback as either 'positive' or 'negative'.")


pydantic_parser = PydanticOutputParser(pydantic_object=FeedBack)

prompt_sentiment=PromptTemplate(
    template='Classify the sentiment of the following feedback from user as either positive or negative: {feedback} in the format: {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)


prompt_positive=PromptTemplate(
    template='Generate a positive response to the following feedback from user: {feedback}',
    input_variables=['feedback']
)

prompt_negative=PromptTemplate(
    template='Generate a negative response to the following feedback from user: {feedback}',
    input_variables=['feedback']
)


parser=StrOutputParser()

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_positive | model | parser),
    (lambda x: x.sentiment == 'negative', prompt_negative | model | parser),
    RunnableLambda(lambda x: "Invalid sentiment classification. Please provide either 'positive' or 'negative' feedback.")
)


classifier_chain = prompt_sentiment | model | pydantic_parser

# print(classifier_chain.invoke({'feedback':'I like this product!'}))


final_chain = classifier_chain | branch_chain

result=final_chain.invoke({'feedback': 'I love the new features in your product! It has made my life so much easier.'})

print("-----------------------------------------ConditionalChainResult-----------------------------------------\n")
print(result)


print("\n-----------------------------------------Execution Graph---------------------------------------\n")
final_chain.get_graph().print_ascii()


