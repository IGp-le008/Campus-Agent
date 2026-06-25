import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

prompt1 = PromptTemplate(
    template="Write 3 points about the topic: {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Write a summary of the following points: {points}",
    input_variables=["points"]
)

parser=StrOutputParser()

chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser)


result=chain.invoke({"topic":"Artificial Intelligence"})

print("------------------------------Runnable Sequence Result---------------------------------")

print(result)

print("------------------------------Execution Graph---------------------------------  ")

chain.get_graph().print_ascii()