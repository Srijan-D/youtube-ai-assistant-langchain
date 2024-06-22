# YouTube AI Assistant with LangChain

This project leverages OpenAI's GPT and LangChain to create an AI assistant that can process YouTube videos. Enter any YouTube URL and ask the assistant questions about the video content, get summaries, and perform various tasks to save time.

## Features

- **Ask Questions**: Interact with the AI to ask questions about the content of YouTube videos.
- **Summarization**: Get concise summaries of YouTube videos.
- **Flexible Interactions**: Use the AI for various tasks related to video content analysis.

## Tech Stack

- **Chains in LangChain**: Chains can maintain state through the addition of Memory. This is particularly useful in LLM projects where context needs to be preserved across multiple interactions or calls. For example, in a conversational AI application like ours, the chain can remember previous queries and use that context to generate more coherent responses.
- **Python**: Programming language.
- **OpenAI GPT**: Provides the natural language processing capabilities, GPT model is being used for querying.
- **LangChain**:
  - **Document Loading**: Extracting transcripts from YouTube videos using `YoutubeLoader`.
  - **Text Splitting**: Dividing large text into manageable chunks using `RecursiveCharacterTextSplitter` for processing by language models.
  - **Embedding Generation**: Converting text into numerical vectors using `OpenAIEmbeddings` that can be used for similarity search.
  - **Vector Storage**: Storing and querying the embeddings efficiently using `FAISS`.
  - **Chat Models and Chains**: Creating an interactive chat model using `ChatOpenAI` and `LLMChain` that can answer questions based on the video transcript.
- **FAISS**: Vector database for efficient similarity search and clustering of video content. Faiss is a library — developed by Facebook AI — that enables efficient similarity search.
- **YouTube API**: For retrieving YouTube video details, content, and transcription.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Srijan-D/youtube-ai-assistant-langchain.git
   cd youtube-ai-assistant-langchain
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```bash
   python youtube-ai-assistant.py
   ```

   1. Go to `youtube_ai_assistant.py` and change the `video_url` at line number 75 to the URL of the YouTube video you want to interact with, change the `query` and change the query to the question you want to ask.

2. Interact with the AI assistant by asking questions or requesting summaries of the video content.

## File Structure

- **assistant.py**: Configures LLMChain to pass OpenAI GPT model and chat template.
- **requirements.txt**: Lists the dependencies required to run the project.
- **youtube-ai-assistant.py**: Main script to interact with the AI assistant.
- **.gitignore**: Specifies files and directories to be ignored by git.

## Contributions

Contributions are welcome! Please fork the repository and submit pull requests.
