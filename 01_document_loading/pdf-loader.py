from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "docs/cs229_lectures/MachineLearning-Lecture01.pdf")
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(len(pages))
print(pages[0].page_content[0:500])
print("-" * 50)
print(pages[0].metadata)
