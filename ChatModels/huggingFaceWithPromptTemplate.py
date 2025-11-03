from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
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

firstPrompt=firstTemplate.invoke({"topic":"blackhole"})
response=chat_model.invoke(firstPrompt)
print(response.content)
secondPrompt=secondTemplate.invoke({"text":response.content})

finalOutput=chat_model.invoke(secondPrompt)

print(finalOutput.content)