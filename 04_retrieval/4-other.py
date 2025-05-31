from langchain_community.retrievers import SVMRetriever, TFIDFRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(
    script_dir, '..', "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
)
loader = PyPDFLoader(pdf_path)
pages = loader.load()

all_page_text = [p.page_content for p in pages]
joined_page_text = " ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(joined_page_text)

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits, embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

print("\nSVM Retriever:")
question = "What are major topics for this class?"
docs_svm = svm_retriever.get_relevant_documents(question)
docs_svm[0]


print("\nTFIDF Retriever:")
question = "what did they say about matlab?"
docs_tfidf = tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]
