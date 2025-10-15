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
#TODO: implement multi-agent network that enables efficient generation of mutiple files with dependencies