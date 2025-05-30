from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(
    script_dir, '..', "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
)
loader = PyPDFLoader(pdf_path)

pages = loader.load()


text_splitter = TokenTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
)

text1 = "foo bar bazzyfoo"

split1 = text_splitter.split_text(text1)
print(split1)


text_splitter = TokenTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
)

docs = text_splitter.split_documents(pages)

print(docs[1])

print("-" * 50)

print(docs[1].page_content)


print(pages[1].metadata)
