from langchain_community.document_loaders import TextLoader

loader = TextLoader("../media/ai_essay.txt")
documents = loader.load()

print("--------------------Page Content--------------------\n")

print(documents[0].page_content)

print("--------------------Metadata--------------------\n")

print(documents[0].metadata)