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
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
import re
from datetime import datetime
from langchain.schema.runnable import RunnablePassthrough
from collections import defaultdict



app = FastAPI()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini" , max_tokens=200)
# Define the input schema for the flight query tool
class FlightQueryInput(BaseModel):
   query: str


def parse_flight_query(query):
    patterns = {
        'departure': r'from\s+([A-Za-z\s]+?)(?:\s+to\b|\s*$)',
        'destination': r'(?:to|for)\s+([A-Za-z\s]+)',
        'date': r'on\s+(\d{4}-\d{2}-\d{2})',
        'price_max': r'under\s+(\d+)',
        'price_min': r'over\s+(\d+)',
        'time': r'at\s+(\d{2}:\d{2})',
        'flight_number': r'\b([A-Z]{2}\d{3,4})\b'  # Matches airline code (2 letters) followed by 3-4 digits
    }
    
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            parsed[key] = match.group(1).strip()
    
    # If no fields matched, check if the query is just a destination
    if not parsed:
        destination_match = re.search(r'^(?:flights?\s+(?:to|for)\s+)?([A-Za-z\s]+)$', query, re.IGNORECASE)
        if destination_match:
            parsed['destination'] = destination_match.group(1).strip()
    
    return parsed

def query_flights(flights, **kwargs):
    
    def flight_matches(flight, criteria):
        return all(str(flight.get(key, '')).lower() == str(value).lower() for key, value in criteria.items())
    return [flight for flight in flights if flight_matches(flight, kwargs)]

def flight_query_tool(query):
    print("Querying for:", query)
    
    if isinstance(query, str):
        parsed_query = parse_flight_query(query)
        print(parsed_query)
    elif isinstance(query, dict):
        parsed_query = query
    else:
        return json.dumps([{"error": "Invalid query format"}])
    
    if not parsed_query:
        return json.dumps([])

    # Use the query_flights function to get matching flights
    matching_flights = query_flights(mock_flights, **parsed_query)

    # Limit to 5 results
    result_flights = matching_flights[:5]


    return json.dumps(result_flights if result_flights else [])
 

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
    You are a helpful AI travel assistant. Your primary task is to help users find flight information, but you can also answer general questions about travel and engage in friendly conversation.

    When users ask about flights, use the FlightQueryTool to search for relevant flight information.
    For general questions or greetings, respond directly without using any tools.

    User Query: {input}

    {agent_scratchpad}
  
    Based on the above information, provide a helpful response to the user's query.
    If you need to search for flights, use the format:
    Action: FlightQueryTool
    Action Input: <relevant search term>

    For general questions or greetings, use the format:
    Final Answer: <your response here>

    Remember:
    1. Only use FlightQueryTool when the user is specifically asking about flights.
    2. For greetings or general questions, respond directly without using any tools.
    3. When using FlightQueryTool, return ONLY the raw JSON output. Do not add any explanations or formatting.
    4. If there are no matching flights, the tool will return an empty array []. Do not add any explanatory text.
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
    output: List[Dict[str, Any]]

def ensure_json_output(result):

    if isinstance(result, dict) and 'output' in result:
        try:
            # Attempt to parse the output as JSON
            json_output = json.loads(result['output'])
             # Determine visualization based on the number of elements
            visualization_type = "flight-list" if len(json_output) > 3 else "flight-details"
            if isinstance(json_output, list):
                if not json_output:  # Empty list
                    return {"output": [{"message": "I'm sorry, but no flights are available matching your criteria."}]}

               

                # It's already a JSON array, return it directly
                print({"output": json_output, "visualization": "flight-details"})
                return {"output": json_output, "visualization": "flight-details"}
            else:
                print({"output": [json_output],"visualization": "flight-list"})
                # If it's not a list, wrap it in a list
                return {"output": [json_output],"visualization": "flight-list"}
        except json.JSONDecodeError:
            # If it's not valid JSON, check for timeout or iteration limit
            if "iteration limit" in result['output'].lower() or "time limit" in result['output'].lower():
                return {"output": [{"error": "I'm sorry, but no flights are available matching your criteria."}]}
            # Otherwise, it's likely a general response
            return {"output": [{"response": result['output']}]}
    else:
        return {"output": [{"error": "Unexpected output format"}]}
# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Create a runnable that ensures JSON output
json_runnable = RunnablePassthrough() | agent_executor | ensure_json_output

@app.get("/query_flights/{query}")
def main(query: str):
    if not query:
        return {"error": "Query is empty"}
    try:
        parsed_query = parse_flight_query(query)
    except Exception as e:
        return {"error": str(e)}
    
    if not parsed_query:
        return {"flights": []}

    try:
        matching_flights = query_flights(mock_flights, **parsed_query)
        result_flights = matching_flights[:5]
        return {"flights": result_flights}
    except Exception as e:
        return {"error": str(e)}
    







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
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)