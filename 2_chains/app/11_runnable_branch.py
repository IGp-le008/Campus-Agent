import os
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

parser=StrOutputParser()

summary_prompt = PromptTemplate(
    template="Write a summary about the topic : {topic}",
    input_variables=["topic"]
)


summary_chain = RunnableSequence(summary_prompt, model, parser)

result = summary_chain.invoke({'topic':'Artificial Intelligence'})


print(result)