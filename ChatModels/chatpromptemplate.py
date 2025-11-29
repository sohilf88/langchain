from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()


history="where is my refund"
chatTemplate=ChatPromptTemplate([
    ("system","you are helpful {domain} expert"),
    
    ("human","explain on {topic}")
])
formatted_prompt = chatTemplate.format_messages(domain="technology", topic="quantum computing")

response=model.invoke(formatted_prompt)

