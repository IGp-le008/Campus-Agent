import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

parser= JsonOutputParser()

prompt_template= PromptTemplate(
    template='Generate name,age and profession of a fictional person \n in {format_instruction}.',
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain= prompt_template | model | parser
result=chain.invoke({})

print("-----------------------------------------JsonParserChainResult-----------------------------------------\n")

print(result)


print("\n-----------------------------------------ExecutionGraph---------------------------------------\n")

chain.get_graph().print_ascii()