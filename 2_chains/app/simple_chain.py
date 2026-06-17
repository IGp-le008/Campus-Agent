import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

prompt = PromptTemplate(
    template='Generate an essay about {topic} in about 150 words.',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'Anthropic'})

print("-----------------------------------------Chain Result-----------------------------------------\n")
print(result)
print("\n-----------------------------------------Execution Graph---------------------------------------\n")
chain.get_graph().print_ascii()
print("\n-----------------------------------------------------------------------------------------------")
