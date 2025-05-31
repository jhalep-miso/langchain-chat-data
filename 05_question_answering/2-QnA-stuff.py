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


question = "What are major topics for this class?"
docs = vectordb.similarity_search(question, k=3)
print(len(docs))


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Prompt
# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

print("\nStuff retrieval QA chain")
# Run chain. It uses stuff by default which stuffs all the documents into the prompt, involving a lot of tokens in a single call.
# If there are a lot of documents, these might no fit inside the context window of the LLM.
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "Is probability a class topic?"

result = qa_chain({"query": question})

print(f"Question: {question}")
print(f"Answer: {result["result"]}")
