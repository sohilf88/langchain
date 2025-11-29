from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Use a chat-compatible model
hf_endpoint = HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mixtral-8x7B-Instruct",
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=512
)# type: ignore

# Wrap as Chat model
huggingFaceModel = ChatHuggingFace(llm=hf_endpoint)

openAiChatModel=ChatOpenAI()
# prompt template

templateForHF=PromptTemplate(
    template="give brief detail about question /n {question}",
    input_variables=["question"]
)

parser=StrOutputParser()

chain = templateForHF |  huggingFaceModel| parser

response=chain.invoke({"question":"nginx rtmp congiration on aws linux 2023 machine and need to convert rtmp into mp4 file with h265 codec and file size splite after 2GB"})

print(response)