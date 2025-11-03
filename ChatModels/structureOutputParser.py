from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
import os
from dotenv import load_dotenv

# Load token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Use a chat-compatible model
hf_endpoint = HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)# type: ignore

# Wrap as Chat model
chat_model = ChatHuggingFace(llm=hf_endpoint)
# create schema
outputStructureSchema=[
    ResponseSchema(name="condition-1",description="some detail about topic"),
    ResponseSchema(name="condition-2",description="some detail about topic"),
    ResponseSchema(name="condition-3",description="some detail about topic")
]
   

# create parser

parser=StructuredOutputParser.from_response_schemas(outputStructureSchema)

# template
template=PromptTemplate(template="give me detail overview of {topic},\n {formatInstruction}",
                        input_variables=["topic"],
                        partial_variables={"formatInstruction":parser.get_format_instructions()})

# prompt=template.invoke({"topic":"blackhole"})
# response=chat_model.invoke(prompt)
# result=parser.parse(response.content)
# print(result)
chain=template | chat_model | parser

response=(chain.invoke({"topic":"open world game"}))

print(response)