import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate

load_dotenv()

instruct_model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)   

claude_model=ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="MihaiPopa-1/Qwen-3-0.6B-Claude-4.7-Opus-Distilled:featherless-ai"
)

prompt1=PromptTemplate(
    template='Generate a short summary from this following refrence text :{reference_text}',
    input_variables=['reference_text']
)

prompt2=PromptTemplate(
    template='Generate 5 questions from this following text :{reference_text}',
    input_variables=['reference_text']
)

prompt3=PromptTemplate(
    template='Generate an exam style question which consists of short passage and questions :\n The paragraph is: \n {short_note} \n and the questions are: \n {questions} \n Include both the passage and questions in the output.',
    input_variables=['short_note','questions']
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'short_note': prompt1 | instruct_model | parser,
    'questions': prompt2 | claude_model | parser
})

merge_chain = prompt3 | instruct_model | parser

chain=parallel_chain | merge_chain

text="""
Topic: The Role of Edge Computing in Modern IoT Ecosystems

Traditional cloud computing architectures rely heavily on centralized data centers to process information gathered by remote endpoints. However, the exponential rise of Internet of Things (IoT) devices has introduced critical bottlenecks, primarily concerning data transmission latency and network bandwidth consumption. To alleviate these constraints, modern architectures increasingly implement edge computing. By shifting computational workloads, data storage, and analytics from centralized cloud servers directly to localized network nodes—such as gateways, routers, or the smart devices themselves—edge computing facilitates near-instantaneous data processing. This proximity significantly minimizes latency, making it vital for time-critical applications like autonomous vehicular navigation and real-time medical monitoring. Furthermore, processing data locally acts as a defensive filter, ensuring that only condensed, high-value metadata is uploaded to the main cloud framework, thereby drastically reducing bandwidth overhead and lowering cloud storage costs.
"""

result=chain.invoke({'reference_text':text})

print("-----------------------------------------ParallelChainResult-----------------------------------------\n")

print(result)

print("\n-----------------------------------------ExecutionGraph---------------------------------------\n")

chain.get_graph().print_ascii()