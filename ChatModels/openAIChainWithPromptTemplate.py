from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# load env
load_dotenv()
# create model 

model=ChatOpenAI()

# create template

template=PromptTemplate(
    template="give me detail overview about {topic}",
    input_variables=["topic"]
)

# second template

second_template=PromptTemplate(
    template="generate summary in 5 lines for below text with numbers \n {text}",
    input_variables=["text"]
)

# create parser
parser=StrOutputParser()
# create chain

chain= template | model | parser | second_template | model | parser

response=chain.invoke({"topic":"sex"})

print(response)