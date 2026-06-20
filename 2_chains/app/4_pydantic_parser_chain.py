import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parser import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)


class PersonInfo(BaseModel):
    name: str =Field(description="The name of the person")
    age: int=Field(description="The age of the person")
    profession: str=Field(description="The profession of the person")

parser=PydanticOutputParser(pydantic_object=PersonInfo)

template=PromptTemplate(
    template='Generate name,age and profession of a fictional person from the country {country_name} in the following format: \n{format_instruction}',
    input_variables=['country_name'],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain=template | model | parser
result=chain.invoke({'country_name':'Nepal'})

print("-----------------------------------------PydanticParserChainResult-----------------------------------------\n")

print(result)

print("\n-----------------------------------------ExecutionGraph---------------------------------------\n")

chain.get_graph().print_ascii()