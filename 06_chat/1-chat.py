from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import getpass
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
docs = vectordb.similarity_search("What are the major topics for this class?", k=3)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Prompt
# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ConversationalRetrievalChain
retriever = vectordb.as_retriever()
condense_question_template = """
    Given the following conversation (between Human and Assistant) and a Follow Up Question, 
    rephrase the Follow Up Question to be a Standalone question, in its original language 
    containing the relevant context from the chat history.
    
    Chat History: {chat_history}
    Follow Up question: {question}
    Standalone question:
"""
condense_question_prompt = PromptTemplate.from_template(condense_question_template)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=condense_question_prompt,
)

question = "Is probability a class topic?"
result = qa({"question": question})
print(f"Question: {question}")
print(f"Answer: {result['answer']}")

# New question
question = "why are those prerequesites needed?"
result = qa({"question": question})
print(f"Question: {question}")
print(f"Answer: {result['answer']}")


# Unrelated question
question = "What is a Prompt Template?"
result = qa({"question": question})
print(f"Question: {question}")
print(f"Answer: {result['answer']}")