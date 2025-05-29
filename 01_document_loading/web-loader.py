from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md"
)
docs = loader.load()
print(len(docs))
print(docs[0].page_content[0:500])
