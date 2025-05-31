from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding_model.embed_query(sentence1)
embedding2 = embedding_model.embed_query(sentence2)
embedding3 = embedding_model.embed_query(sentence3)


print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))
print(np.dot(embedding2, embedding3))
