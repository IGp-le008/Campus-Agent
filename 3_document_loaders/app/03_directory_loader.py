import os
import numpy as np
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
Write 5 key points in short from the following text:

{text}
""",
    input_variables=["text"]
)

loader = DirectoryLoader(
    path="../media",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

np_docs = np.asarray(docs)

if len(np_docs) == 0:
    print("No PDF files found in the '../media' directory.")
    exit()

print("\n__________ Metadata Of Page 1 __________\n")
print(np_docs[0].metadata)

print("\n-------------------- LLM Summarized Output --------------------\n")

text_content = np_docs[0].page_content

chain = prompt | model | parser

result = chain.invoke({
    "text": text_content
})

print(result)

print("--------------------LLMExecutionGraph--------------------")
chain.get_graph().print_ascii()