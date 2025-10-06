from openai import OpenAI
import os
import json

# initialize client'
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# provided problem context of problem
context = {
    "agent_type": 1,
    "neighbour_type": {"type_1": 3, "type_2": 5},
    "position": "Agent is in a lattice with eight neighbours, which is either type 1 or 2.",
}

# example prompting template (no literature search right now)
system_prompt = """
You are a helpful research assistant who specializes in constructing social simulation models based on provided context.
You will be provided with a context describing the agent's environment and characteristics.
Your task is to analyze the context and generate a set of if–then rules that define the agent's behavior in the simulation.
You should think step-by-step and ensure that the rules are clear, concise, and logically sound.
Output strictly in JSON:
{
  "rules": [
    "IF condition THEN action",
    "..."
  ],
  "my_current_situation": "Describe the situation of the agent based on the provided context.",
  "next_action_based_on_rules": "Describe the action the agent will take based on the rules."
}
Ensure your response is valid JSON that can be parsed by a JSON parser.
"""

user_prompt = f"""
Imagine you are an agent in a Shelling segragation model. 
The agents in this model can be of two types: type 1 and type 2.
The agent is situated in a lattice where it has eight neighbours, which can be either type 1 or type 2.
You are an agent of type {context['agent_type']}.
You have {context['neighbour_type']['type_1']} neighbours of type 1 and {context['neighbour_type']['type_2']} neighbours of type 2.
Think step-by-step and determine whether you are happy or unhappy, and what steps you are going to take next.
Agents are usually happy when a sufficient proportion of their neighbors are of the same type (e.g., 50% or more). 
You can adjust this threshold if reasoning suggests so.
Translate this reasoning into a small set of explicit if–then rules.
"""


def literature_search(query: str) -> str:  # TODO: to be implemented later
    """
    A tool that allows AI agent to search for relevant literature through API call
    """
    return "No literature yet"


# initialize LLM model
llm_model = client.chat.completions.create(
    model="gpt-4o-mini",  # start with a cheap model
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ],
)

# export the output response
response = llm_model.choices[0].message.content
try:
    rules = json.loads(response)
    print(json.dumps(rules, indent=2))
except json.JSONDecodeError:
    print("LLM output not valid JSON, here’s raw output:")
    print(response)
