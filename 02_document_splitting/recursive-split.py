from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

print(len(some_text))

char_splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=0, separator=" ")
recur_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
)

print(char_splitter.split_text(some_text))
print(recur_splitter.split_text(some_text))

# Reducing the chunk size and include a period in separators
recur_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""],
)
print()
print(recur_splitter.split_text(some_text))


recur_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=0, separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

print()
print(recur_splitter.split_text(some_text))
