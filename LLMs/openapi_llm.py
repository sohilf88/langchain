from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()



client=ChatGoogleGenerativeAI( model="gemini-2.5-flash", temperature=0.9 )

# userQueryHistory=[]
# while True:
#     user_input=input("you:")
#     userQueryHistory.append(user_input)
#     if (user_input=="stop"):
#         break
#     response=client.invoke(userQueryHistory)
#     userQueryHistory.append(response.content)
#     print("AI: ", response.content)