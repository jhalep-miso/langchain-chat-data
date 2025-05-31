from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import getpass
from langchain_chroma import Chroma
import langchain

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# Vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(
    script_dir, "..", "docs/chroma_db"
)  # using stored data from 03_vectorstores_and_embeddings/2-vectorstores.py


embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Another approach for improving the quality of retrieved docs is compression.
# Information most relevant to a query may be buried in a document with a lot of irrelevant text.
# Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
# Contextual compression is meant to fix this.

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectordb.as_retriever()
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)


# Combining techniques
print("\nCombining techniques")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectordb.as_retriever(search_type="mmr")
)

compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)
