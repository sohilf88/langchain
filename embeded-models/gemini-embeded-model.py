from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

content=["what is captial of india","Hello there","How are you"]
embeded=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

response=embeded.embed_documents(content)
print(response)