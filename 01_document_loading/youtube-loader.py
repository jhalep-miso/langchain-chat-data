from langchain_community.document_loaders.generic import (
    GenericLoader,
    FileSystemBlobLoader,
)
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader
import os

url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "..", "docs/youtube/")
loader = GenericLoader(
    # YoutubeAudioLoader([url],save_dir),  # fetch from youtube
    FileSystemBlobLoader(save_dir, glob="*.m4a"),  # fetch locally
    FasterWhisperParser(),
)
docs = loader.load()

print(len(docs))

print(docs[0].page_content[0:500])
