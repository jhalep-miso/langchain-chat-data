from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(
    script_dir, '..', "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
)
loader = PyPDFLoader(pdf_path)

pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len
)

docs = text_splitter.split_documents(pages)

print(len(pages))
print(len(docs))
