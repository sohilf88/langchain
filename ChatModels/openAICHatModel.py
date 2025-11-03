
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import List,Annotated
load_dotenv()

chatmodel = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # or whichever model you want

class OutputResponse(BaseModel):
    success:Annotated[bool,Field(description="it is either True or False",default=False)]
    result: Annotated[str,Field(description="result of output in string format")]

# structured output wrapper
structured_model = chatmodel.with_structured_output(OutputResponse)

response = structured_model.invoke("who killed Gandhi and why")
print(response)
