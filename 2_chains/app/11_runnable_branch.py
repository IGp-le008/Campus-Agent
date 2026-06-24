import os
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
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

parser = StrOutputParser()

summary_prompt = PromptTemplate(
    template="Write a detailed summary on the topic: {topic}",
    input_variables=["topic"]
)

short_summary_prompt = PromptTemplate(
    template="The summary is too long! Reduce its length significantly: {summary}",
    input_variables=['summary']
)

def word_count(text):
    return len(text.split())

# Chains
summary_chain = summary_prompt | model | parser
short_summary_chain = short_summary_prompt | model | parser

# Branch logic - Route to the shortening chain if word count exceeds 500
branch_chain = RunnableBranch(
    (lambda x: word_count(x) > 500, short_summary_chain),
    RunnablePassthrough()
)

# Parallel execution map
parallel_chain = RunnableParallel({
    'initial_summary': RunnablePassthrough(),
    're-summarized': branch_chain
})

# Sequential connection using modern LCEL pipe operators
final_chain = summary_chain | parallel_chain

# Run
result = final_chain.invoke("Artificial Intelligence")

# Corrected dictionary lookup keys with string quotes
print(f"Initial Summary (Words: {word_count(result['initial_summary'])}):\n{result['initial_summary']}\n")
print(f"Re-Summarized (Words: {word_count(result['re-summarized'])}):\n{result['re-summarized']}")