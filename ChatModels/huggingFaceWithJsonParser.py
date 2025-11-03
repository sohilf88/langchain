from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
# get api key from env file
api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# create hugginFace endpoint
hf_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.5,
    huggingfacehub_api_token=api_key
)# type: ignore

# create chat model and add endpoint
model=ChatHuggingFace(llm=hf_endpoint)

# create parser
parser=JsonOutputParser()

# prompt/input to chat model
template=PromptTemplate(
    template="generate an user detail with username, email and password {format_instruction}",input_variables=[],
    partial_variables={
                        "format_instruction":parser.get_format_instructions()
                      })
prompt=template.format()

response=model.invoke(prompt)
result=parser.parse(response.content)#type:ignore
# chain= template | model | parser
# result=chain.invoke({})
print(type(result))
print(result)