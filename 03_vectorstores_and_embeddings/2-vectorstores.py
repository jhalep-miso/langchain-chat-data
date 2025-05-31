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

script_dir = os.path.dirname(os.path.abspath(__file__))
get_path = lambda x: os.path.abspath(os.path.join(script_dir, '..', x))

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader(get_path("docs/cs229_lectures/MachineLearning-Lecture01.pdf")),
    PyPDFLoader(get_path("docs/cs229_lectures/MachineLearning-Lecture01.pdf")),
    PyPDFLoader(get_path("docs/cs229_lectures/MachineLearning-Lecture02.pdf")),
    PyPDFLoader(get_path("docs/cs229_lectures/MachineLearning-Lecture03.pdf")),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

splits = text_splitter.split_documents(docs)
print(len(splits))

# Vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(script_dir, "..", "docs/chroma_db")


embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory,
)

print(vectordb._collection.count())

# Similarity search
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question, k=3)

print(len(docs))

# Failure models

## Edge case 1: duplicate chunks
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question, k=5)

print("Edge case 1: duplicate chunks")
print(question)
# we're getting duplicate chunks (because of the duplicate MachineLearning-Lecture01.pdf in the index).
# Semantic search fetches all similar documents, but does not enforce diversity.
print(docs[0].metadata)
print(docs[1].metadata)


print("Edge case 2: references")
# The question below asks a question about the third lecture, but includes results from other lectures as well.
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question, k=5)
for doc in docs:
    print(doc.metadata)
