from langchain_text_splitters import CharacterTextSplitter
text = """
Artificial Intelligence (AI) is one of the most significant technological developments of the modern era. It refers to the ability of computers and machines to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, decision-making, understanding language, and recognizing images or patterns. AI has transformed the way people interact with technology and has become an essential part of daily life. From virtual assistants on smartphones to recommendation systems used by online shopping platforms, AI is making processes faster, more efficient, and more personalized.

The goal of artificial intelligence is not simply to replace human effort but to assist people in completing tasks more accurately and efficiently. Researchers and engineers continue to develop AI systems that can analyze large amounts of data, identify patterns, and make informed decisions with minimal human intervention. As computing power and data availability continue to increase, AI is expected to play an even greater role in society.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap= 0
)

result = splitter.split_text(text)
print(result)
