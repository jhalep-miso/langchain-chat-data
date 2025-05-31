from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import getpass
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

question = "Is probability a class topic?"
result = qa_chain({"query": question})

print(result["result"])
