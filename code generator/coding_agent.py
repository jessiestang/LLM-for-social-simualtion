# implement the coding agent
import os
from openai import OpenAI
import json

class CodingAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # set your API key in environment variable
        self.model_name = model_name  # default model

    
    # first trial: generate only one python file based on problem definition
    def code_generation(self, json_path):
        """
        This coding agent will generate a python file written mainly with MESA framework
        based on the problem definition provided by the model construct assistant on
        the previous step.
        """
        # import json file
        with open(json_path, "r") as f:
            conceptual_model = json.load(f)

        # prompting the LLM
        system_prompt = """
        You are a research assistant in computational social science who specializes in python programming with MESA framework.
        You will be given a json-format problem definition.
        Your jon is to generate a python file that implements the model described in the problem definition.
        The file should be as complete as possible, with all necessary imports, class definitions, and functions.
        Provide each file clearly seperated by markers like:
        [BEGIN FILE: model.py]
        ...
        [END FILE: model.py]
        The code should be clean, well-commented, modular and runnable.
        """

        user_prompt = f"""
        Please write up the code based on this conceptual model: {conceptual_model}
        Make sure to follow the instructions in the system prompt.
        """

        # call the LLM
        LLM_model = self.client.chat.completions.create(
            model=self.model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )

        # parse the responses
        code = LLM_model.choices[0].message.content
        with open("generated_model.py", "w") as f: # export to a python file
            f.write(code)

        return code

class SubCodeAgents:
    def __init__(self, module_name, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # set your API key in environment variable
        self.model_name = model_name  # default model
        self.module_name = module_name # the specific module this sub-agent is responsible for
    """
    The sub-coding agents listen to the instructions from the master agent.
    Each sub-coding agent is responsible for generating a specific part of the codebase.
    They will receive a detailed description of the component they need to implement,
    along with any relevant context from the overall model, and a list of dependencies on other components from the master agent.
    They will send their completed code back to the master agent for integration.
    """
    def code_generator(self, shared_instructions, conceptual_model):
        """
        This function generates code for a specific component based on shared instructions and the overall conceptual model.
        The files to be generated: agent.py, environment.py, model.py, run.py, visualization.py
        """
        # parse the json insturctions and model
        shared_instructions = json.dumps(shared_instructions, indent=2)
        conceptual_model = json.dumps(conceptual_model, indent=2)

        # prompting the LLM
        system_prompt = """
        You are a coding assistant specialized in generating Python code for MESA-based agent-based models.
        Your task is to generate code for a specific component of the model.
        You will receive shared instructions and a conceptual model as input from the master agent.
        Please stick strictly to the shared_instructions, which state which class name, variable name, function name, parameters to use.
        The shared_instructions also specify which package you can import.
        Do not invent any new names or parameters, to ensure the compatibility with other code files.
        The code file should be clean, well-commented, modular and runnable.
        """

        user_prompt = f"""
        Please write up the MESA code for the {self.module_name} component based on the following shared instructions: {shared_instructions}.
        The overall conceptual model is: {conceptual_model}.
        Please stick strictly to {shared_instructions}. Do not invent any new names or parameters.
        """

        # call the LLM
        LLM_model = self.client.chat.completuons.create(
            model = self.model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature = 0.2,
            max_tokens = 1500
        )

        # parse the responses
        generated_code = LLM_model.choices[0].message.content

        return generated_code  # return to the master agent; master agent will integrate all code files and output the final model
    
    #TODO: implement the master agent

