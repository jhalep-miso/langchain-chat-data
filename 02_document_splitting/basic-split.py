from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

chunk_size = 26
chunk_overlap = 4

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

char_splitter = CharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Recursive split examples
print(
    f"Recursive split examples with chunk size {chunk_size} and overlap {chunk_overlap}"
)
text1 = "abcdefghijklmnopqrstuvwxyz"
recursive_chunks = recursive_splitter.split_text(text1)
print(
    {
        "text1": text1,
        "recursive_chunks": recursive_chunks,
    }
)


text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
recursive_chunks = recursive_splitter.split_text(text2)
print(
    {
        "text2": text2,
        "recursive_chunks": recursive_chunks,
    }
)

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
recursive_chunks = recursive_splitter.split_text(text3)
print(
    {
        "text3": text3,
        "recursive_chunks": recursive_chunks,
    }
)

# Char split
print(f"Char split examples with chunk size {chunk_size} and overlap {chunk_overlap}")
char_chunks = char_splitter.split_text(text3)  # Default separator in \n\n
print(
    {
        "text3": text3,
        "char_chunks": char_chunks,
    }
)

char_splitter = CharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" "
)

print('changing separator to " "')
char_chunks = char_splitter.split_text(text3)
print(
    {
        "text3": text3,
        "char_chunks": char_chunks,
    }
)
