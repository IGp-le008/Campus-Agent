import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

prompt = PromptTemplate(
    template="Write in short about the topic: {topic} in about 30 words",
    input_variables=["topic"]
)

parser=StrOutputParser()

def word_count(text):
    return len(text.split())


summary_chain= RunnableSequence(prompt,model,parser)

word_count_chain = RunnableLambda(word_count)

final_chain = RunnableParallel({
    "summary" : summary_chain,
    "word_count" : RunnableSequence(summary_chain, word_count_chain)
})
result=final_chain.invoke({"topic":"Artificial Intelligence"})


print("------------------------------Runnable Lambda Result---------------------------------")
print("Summary: ", result["summary"])
print("Word Count: ", result["word_count"])


print("------------------------------Execution Graph---------------------------------  ")
final_chain.get_graph().print_ascii()