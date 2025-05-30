# Context aware splitting
# Chunking aims to keep text with common context together.
# A text splitting often uses sentences or other delimiters to keep related text together but many documents (such as Markdown) have structure (headers) that can be explicitly used in splitting.
# We can use MarkdownHeaderTextSplitter to preserve header metadata in our chunks, as show below.

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import NotionDirectoryLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "..", "docs/Notion_DB")
loader = NotionDirectoryLoader(save_dir)
docs = loader.load()


markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits[0])
print(md_header_splits[1])

# from notion files
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

md_header_splits = markdown_splitter.split_text(txt)

print("-" * 50)
print(md_header_splits[0])
