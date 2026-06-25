import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnableSequence
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
    template="Write a short instagram post about the topic: {instagram_topic} in about 50 words", 
    input_variables=["instagram_topic"]
)

prompt2 = PromptTemplate(
    template="Write a short linkedin post about the topic: {linkedin_topic} in about 50 words", 
    input_variables=["linkedin_topic"]
)

prompt3 = PromptTemplate(
    template="List out 3 main points from each of the following text: \n Instagram: {instagram_post} \n LinkedIn: {linkedin_post}",    
    input_variables=["instagram_post", "linkedin_post"]
)

parser = StrOutputParser()

parallel_chain= RunnableParallel({
    "instagram_post": RunnableSequence(prompt1, model, StrOutputParser()),
    "linkedin_post": RunnableSequence(prompt2, model, StrOutputParser())
})

summary_chain = RunnableSequence(prompt3, model, parser)

final_chain = RunnableParallel({
    "post":parallel_chain,
    "summary":RunnableSequence(parallel_chain, summary_chain)
})

  

result = final_chain.invoke({
    "instagram_topic": "Importance of santitary hygiene in daily life",
    "linkedin_topic": "The benefits of meditation for mental health"
})

print("------------------------------Intermediate LLM Post Result---------------------------------")

print(f"#Instagram Post: {result['post']['instagram_post']}\n")
print(f"#LinkedIn Post: {result['post']['linkedin_post']}")


print("------------------------------Runnable Parallel (Post) Result---------------------------------")

print(result["summary"])

print("------------------------------Execution Graph---------------------------------  ")

final_chain.get_graph().print_ascii()

