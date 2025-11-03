from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
# code is corret but hugging face api is not reliable and most of time won't work
# Load .env file
load_dotenv()

# Fetch the Hugging Face token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # ensure this matches your .env key
print("Token loaded:", bool(hf_token))

prompt=ChatPromptTemplate.from_messages([
    ("system","you are helpful AI Assistant"),
    ("user",'{question}')
])
formatted_prompt = prompt.format_messages(question="what is inbound and outbound in Checkpoint firewall")
# Initialize the endpoint
hf = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    
    task="text-generation",
    huggingfacehub_api_token=hf_token  # required for private/gated models
) # type: ignore

# Generate text
response = hf.invoke(formatted_prompt)
print(response)
