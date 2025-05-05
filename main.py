from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
df = pd.read_csv("data.csv")

api_key=os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(    
            google_api_key=api_key, 
            model="gemini-1.5-pro",
            temprature=0
            )


agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
    )

# question ="Summarize the data?"
# response=agent.invoke(question)
def generate_response(question):
    return agent.invoke(question)['output']
