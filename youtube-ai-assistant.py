from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


video_url="https://www.youtube.com/watch?v=jvqFAi7vkBc"


def create_db_from_video(video_url):
    loader=YoutubeLoader(video_url)
    transcript=loader.load()
    
    # now we need to split the transcript into chunks as the model gpt-4 can only handle 4096 tokens at a time
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #chunk_size=1000: This sets the size of each chunk to 1000 characters.
# chunk_overlap=100: This sets the overlap between consecutive chunks to 100 characters.
# This setup means each chunk will be 1000 characters long, but the last 100 characters of one chunk will also be the first 100 characters of the next chunk. This overlapping helps preserve the context that might be lost at the boundaries of the chunks.
    docs=text_splitter.split_documents(transcript) # this will return a list of documents
    
    db=FAISS.from_documents(docs, embeddings) #this will create a vector store from the documents it is facebooks implementation of the approximate nearest neighbor search algorithm
    return db


