from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import List
import os
from dotenv import load_dotenv

# Load token
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
chat_model = ChatHuggingFace(llm=hf_endpoint)
# create schema
class PersonSchema(BaseModel):
    name:str=Field(description="name of person")
    city:str=Field(description="city of person where it live")
    age:int=Field(gt=20,lt=50,description="age of person must be greater than 20 and less than 50")
    address:str=Field(description="address of person")
    married:bool=Field(description="person married or not in True or False")
    country:str=Field(description="country Name")

class Person(BaseModel):
    persons:List[PersonSchema]
    # parser
parser=PydanticOutputParser(pydantic_object=Person)
    # create prompt with Template

template=PromptTemplate(
        template="generate {number_of_person} persons detail live in country {countryName} \n {field_generator}",
        partial_variables={"field_generator":parser.get_format_instructions()},
        input_variables=["countryName","number_of_person"],
        
    )

chain=template |chat_model | parser

countries=["india","pakistan","us","austria","russia"]
result=chain.invoke({"countryName":countries,"number_of_person":100})

print(result.model_dump_json(indent=2))