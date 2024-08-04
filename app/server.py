from fastapi import FastAPI
from langchain.agents import AgentExecutor, Tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import AgentAction, AgentFinish
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel
import json
import random
from data import mock_flights
from typing import Any
from langchain_openai import ChatOpenAI

app = FastAPI()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")


# Custom tool for flight queries
def flight_query_tool(query):
    relevant_flights = []
    for flight in mock_flights:
        if query.lower() in flight['departure'].lower() or query.lower() in flight['destination'].lower() or query in flight['date']:
            flight_with_number = flight.copy()
            flight_with_number['flight-number'] = random.randint(100, 9999)
            relevant_flights.append(flight_with_number)
    
    return json.dumps({
        "visualization": "flight-list",
        "data": relevant_flights[:5]  # Limit to 5 flights for brevity
    })

tools = [
    Tool(
        name="FlightQueryTool",
        func=flight_query_tool,
        description="Useful for querying flight information based on departure, destination, or date."
    )
]

# Agent prompt
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a helpful AI travel assistant. Your task is to help users find flight information.
    Use the FlightQueryTool to search for relevant flights based on the user's query.

    User Query: {input}

    {agent_scratchpad}
  

    Based on the above information, provide a helpful response to the user's query.
    If you need to use a tool, use the format:
    Action: FlightQueryTool
    Action Input: <relevant search term>

    If you have a final response for the user, use the format:
    Final Answer: <your response here>
    """
)

# Create the agent
# agent = create_react_agent(llm, tools, prompt)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

# Agent executor
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor = AgentExecutor(agent=agent, tools=tools,handle_parsing_errors=True)

# Add LangServe route
add_routes(
    app, 
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ), 
    path="/flight-planner"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)