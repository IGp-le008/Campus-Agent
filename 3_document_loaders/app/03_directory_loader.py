from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import numpy as np

loader = DirectoryLoader(
    path = '../media',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()

np_docs = np.asarray(docs)

print(" \n__________Meta Data__________\n ")

print(np_docs[0])