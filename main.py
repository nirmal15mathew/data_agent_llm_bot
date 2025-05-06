from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from google.api_core.exceptions import ResourceExhausted
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
df = pd.read_csv("data.csv")

api_key=os.getenv("GOOGLE_API_KEY")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# llm = ChatGoogleGenerativeAI(    
#             google_api_key=api_key, 
#             model="gemini-1.5-pro",
#             temprature=0
#             )
llm = ChatOpenAI(
    model="meta-llama/llama-4-maverick:free",  # or any other supported model
    temperature=0
)
# llm = Ollama(model="mistral:latest")


agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
    prefix="This dataset contains the house prices in different counties in the us. You are a real estate financial expert. Use it to answer questions more accurately. Never give an output as code to the user. Only provide analysis. Provide explicit data only if asked. If you are unable to answer due to an error or insufficient data, state such in the response and stop.",
    handle_parsing_errors=True
    )

# question ="Summarize the data?"
# response=agent.invoke(question)
def generate_response(question):
     # Get chat history as messages
    history = memory.load_memory_variables({})["chat_history"]
    
    # Combine chat history + new user input
    full_input = {"input": question, "chat_history": history}
    
    try:
        # Run the agent
        response = agent.invoke(full_input)

        # Update memory
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response['output'])

        return response['output']
    except ResourceExhausted as e:
        return 'Qouta Reached. Try again later'
        
if __name__ == "__main__":
    q = input("Enter query: ")
    print(generate_response(q))