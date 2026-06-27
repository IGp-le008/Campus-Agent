# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../media/ai_essay.pdf")

docs = loader.load()

print("----------------------Metadata--------------------\n")

print(docs[0].metadata)

print("----------------------Page Content--------------------\n")

print(docs[0].page_content)

print("----------------------Total Pages--------------------\n")

print(len(docs))