from fastapi import FastAPI
from langchain.agents import AgentExecutor, Tool
from langchain.tools.base import StructuredTool
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
import re
from datetime import datetime
from langchain.schema.runnable import RunnablePassthrough

app = FastAPI()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini" , max_tokens=100)
# Define the input schema for the flight query tool
class FlightQueryInput(BaseModel):
    query: str

def parse_flight_query(query):
    patterns = {
        'departure': r'from\s+([A-Za-z\s]+)',
        'destination': r'to\s+([A-Za-z\s]+)',
        'date': r'on\s+(\d{4}-\d{2}-\d{2})',
        'price_max': r'under\s+(\d+)',
        'price_min': r'over\s+(\d+)',
        'time': r'at\s+(\d{2}:\d{2})',
    }
    
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, query)
        if match:
            parsed[key] = match.group(1)
    
    return parsed


def flight_query_tool(query):
    print("Querying for:", query)
    parsed_query = parse_flight_query(query)
    relevant_flights = []

    for flight in mock_flights:
        matches = True
        for key, value in parsed_query.items():
            if key == 'price_max' and flight['price'] > int(value):
                matches = False
                break
            elif key == 'price_min' and flight['price'] < int(value):
                matches = False
                break
            elif key in flight and str(flight[key]).lower() != value.lower():
                matches = False
                break
        
        if matches:
            flight_with_number = flight.copy()
            flight_with_number['flight-number'] = random.randint(100, 9999)
            relevant_flights.append(flight_with_number)

    return json.dumps(relevant_flights[:5])

tools = [
    StructuredTool(
        name="FlightQueryTool",
        func=flight_query_tool,
        description="Useful for querying flight information based on departure, destination, or date.",
        args_schema=FlightQueryInput
    )
]

# Agent prompt
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a helpful AI travel assistant. Your task is to help users find flight information.
    Use the FlightQueryTool to search for relevant flights based on the user's query.
    You can also use the agent_scratchpad to store any intermediate information that you need to keep track of.
    IMPORTANT: Use the tools only when necessary, basically when you need to search for flights based on the user's query.

    User Query: {input}

    {agent_scratchpad}
  
    Based on the above information, provide a helpful response to the user's query.
    If you need to use a tool, use the format:
    Action: FlightQueryTool
    Action Input: <relevant search term>

    If you have a final response for the user, use the format:
    Final Answer: <your response here>

    IMPORTANT: Any time you use FlightQueryTool , Your final answer must be a valid JSON string containing an array of flight objects. 
    If there are no matching flights, return No flights available right now. 
    Do not include any explanatory text outside the JSON structure.
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

# class Input(BaseModel):
#     input: str

# # Modify the Output class to expect a JSON string
# class Output(BaseModel):
#     output: str

# def ensure_json_output(result):
#     if isinstance(result, dict) and 'output' in result:
#         try:
#             # Attempt to parse the output as JSON
#             json_output = json.loads(result['output'])
#             # Ensure it's an array
#             if not isinstance(json_output, list):
#                 json_output = [json_output]
#             return json.dumps(json_output)
#         except json.JSONDecodeError:
#             # If it's not valid JSON, wrap it in an array and return
#             return json.dumps([{"error": "Invalid JSON", "original_output": result['output']}])
#     else:
#         return json.dumps([{"error": "Unexpected output format"}])

# # Create the agent executor
# agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# # Create a runnable that ensures JSON output
# json_runnable = RunnablePassthrough() | agent_executor | ensure_json_output

# # Add LangServe route with the modified executor
# add_routes(
#     app, 
#     json_runnable.with_types(input_type=Input, output_type=Output).with_config(
#         {"run_name": "agent"}
#     ), 
#     path="/flight_planner"
# )

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: str

def ensure_json_output(result):
    if isinstance(result, dict) and 'output' in result:
        try:
            # Attempt to parse the output as JSON
            json_output = json.loads(result['output'])
            # Ensure it's an array
            if not isinstance(json_output, list):
                json_output = [json_output]
            return {"output": json.dumps(json_output)}
        except json.JSONDecodeError:
            # If it's not valid JSON, wrap it in an array and return
            return {"output": json.dumps([{"error": "Invalid JSON", "original_output": result['output']}])}
    else:
        return {"output": json.dumps([{"error": "Unexpected output format"}])}

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Create a runnable that ensures JSON output
json_runnable = RunnablePassthrough() | agent_executor | ensure_json_output

# Add LangServe route with the modified executor
add_routes(
    app, 
    json_runnable.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ), 
    path="/flight_planner"
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)