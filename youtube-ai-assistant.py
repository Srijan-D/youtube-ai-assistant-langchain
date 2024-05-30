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

def query_response(db, query,k=4):

    docs=db.similarity_search(query,k=k) #get top k similar documents to the query
    page_content=" ".join([d.page_content for d in docs]) # each document has a page content attribute which is the text of the document
    
    chat=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.2)
    
      # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages( 
    # Combines both the system and human message prompts into a single template.
    [system_message_prompt, human_message_prompt]
    )   
        
    chain = LLMChain(llm=chat, prompt=chat_prompt )
#llm=chat: This parameter specifies the language model to use. In this case, it is configured to use the ChatOpenAI model with specific settings (e.g., gpt-3.5-turbo-16k and a temperature of 0.2).
# prompt=chat_prompt: This parameter specifies the prompt template to use. Here, it combines the system message prompt and the human message prompt into a single chat_prompt.

    response = chain.run(question=query, docs=page_content)
    response = response.replace("\n", "")
    return response, docs

video_url = "https://www.youtube.com/watch?v=jvqFAi7vkBc"
db = create_db_from_video(video_url)

query = "what is this video about?"
response, docs = query_response(db, query)
print(textwrap.fill(response, width=50))    