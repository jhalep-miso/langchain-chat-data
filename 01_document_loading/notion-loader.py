from langchain_community.document_loaders import NotionDirectoryLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "..", "docs/Notion_DB")
loader = NotionDirectoryLoader(save_dir)
docs = loader.load()

print(len(docs))
print(docs[0].page_content[0:500])
