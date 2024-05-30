from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate


load_dotenv(find_dotenv())


llm = OpenAI(model_name="text-davinci-003")
prompt = "Write a poem about python and ai"
print(llm(prompt))


# --------------------------------------------------------------
# Prompt Templates: Manage prompts for LLMs
# --------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

prompt.format(product="Smart Apps using Large Language Models (LLMs)")

llm = OpenAI()


