from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model=ChatOpenAI()

# promptTemplate=ChatPromptTemplate([
    
# ])
message=[
    SystemMessage(content="you are helpful assistant"),
    HumanMessage(content="give me difference between langchain and langraph")
]

# response=model.invoke(message)

# message.append(AIMessage(content=response.content))

# print(message)
while True:
    userInput=input("you: ")
    if userInput=="exit":
        break
    message.append(HumanMessage(content=userInput))
    result=model.invoke(message)
    message.append(AIMessage(content=result.content))
    print("AI ",result.content)
# dynamic message conversion using ChatpromptTemplate