from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import getpass
from langchain_chroma import Chroma

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(
    script_dir, "..", "docs/chroma_db"
)  # using stored data from 03_vectorstores_and_embeddings/2-vectorstores.py


embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]


smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"
smalldb.similarity_search(question, k=2)
smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3)

# Addressing diversity: Max Marginal Relevance
# Maximum marginal relevance strives to achieve both relevance to the query and diversity among the results.
question = "what did they say about matlab?"
print(f"Addressing diversity:\nQuestion: {question}")
print("\nSimilarity Search:")
docs_ss = vectordb.similarity_search(question, k=3)
print(docs_ss[0].page_content[:100])
print(docs_ss[1].page_content[:100])

print("\nMax Marginal Relevance:")
docs_mmr = vectordb.max_marginal_relevance_search(question, k=3)
print(docs_mmr[0].page_content[:100])
print(docs_mmr[1].page_content[:100])

# Addressing specificity: working with metadata
#  many vectorstores support operations on metadata; metadata provides context for each embedded chunk.
pdf_path = os.path.abspath(
    os.path.join(script_dir, '..', "docs/cs229_lectures/MachineLearning-Lecture03.pdf")
)
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source": pdf_path},
)

print(len(docs))

for d in docs:
    print(d.metadata)

print(pdf_path)
