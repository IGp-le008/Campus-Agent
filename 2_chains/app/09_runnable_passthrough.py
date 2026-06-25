import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

point_chain= RunnableSequence(prompt1,model,parser)

parallel_chain= RunnableParallel({
    "points":RunnablePassthrough(),
    "summary":RunnableSequence(prompt2,model,parser)
})

final_chain= RunnableSequence(point_chain,parallel_chain)

result=final_chain.invoke({"topic":"Artificial Intelligence"})

print("------------------------------Runnable Passthrough Result---------------------------------")

print("----------Points----------")

print("Points: ", result["points"])

print("----------Summary----------")

print("Summary: ", result["summary"])

print("------------------------------Execution Graph---------------------------------  ")
final_chain.get_graph().print_ascii()