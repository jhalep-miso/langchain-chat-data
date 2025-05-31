from langchain.retrievers.self_query.base import SelfQueryRetriever
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

# Vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(
    script_dir, "..", "docs/chroma_db"
)  # using stored data from 03_vectorstores_and_embeddings/2-vectorstores.py


embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


metadata_field_info = [
    # AttributeInfo(
    #     name="source",
    #     description="The lecture the chunk is from, should be one of `<some-dir>/docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `<some-dir>/docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `<some-dir>/docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
    #     type="string",
    # ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]
document_content_description = "Lecture notes"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

question = "what did they say about regression in the third lecture?"

langchain.debug = True
docs = retriever.get_relevant_documents(question)
langchain.debug = False

print(len(docs))

for doc in docs:
    print(doc.metadata)
