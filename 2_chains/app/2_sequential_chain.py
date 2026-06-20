import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
    
model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="Qwen/Qwen2.5-7B-Instruct"
)


template1=PromptTemplate(
        template='Generate a paragraph about {topic} in about 150 words',
        input_variables=['topic']
)

template2=PromptTemplate(
        template='Generate a 5 point summary from the following text: \n {text}',
        input_variables={'text'}
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser


result=chain.invoke({'topic':'ChatGPT'})


print("-----------------Paragraph--------------\n",result)


print("-----------------Graph------------------\n")

chain.get_graph().print_ascii()

