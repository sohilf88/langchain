from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Use a chat-compatible model
hf_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)# type: ignore

# Wrap as Chat model
chat_model = ChatHuggingFace(llm=hf_endpoint)

# Create first prompt
firstTemplate=PromptTemplate(
    template="kindly provide detailed on {topic}",
    input_variables=["topic"]
)
# second prompt
secondTemplate=PromptTemplate(
    template="summerize given text in 10 lines with number \n {text}",
    input_variables=["text"]
)
parser=StrOutputParser()

chains=firstTemplate | chat_model | parser | secondTemplate | chat_model | parser

response=chains.invoke({"topic":"pubg game"})

print(response)