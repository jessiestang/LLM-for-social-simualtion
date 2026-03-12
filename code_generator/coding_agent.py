# implement the coding agent
import os
from openai import OpenAI
import json
import re
import traceback
import tempfile
import subprocess
import sys

class CodingAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # set your API key in environment variable
        self.model_name = model_name  # default model

    
    # first trial: generate only one python file based on problem definition
    def code_generation(self, json_path:str, user_requirements:str):
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
        You are an expert in computational social science who specializes in python programming with MESA framework.
        You will be given a json-format problem definition.
        Your job is to generate a python file that implements the model described in the problem definition.
        The file should be as complete as possible, with all necessary imports, class definitions, and functions.
        You file should contain at least following components:
        - Agent class that adds, selects, shuffles agents and defines their attributes and methods
        - Model class that activates agents, defines the event scheduler, collects agent and system data, and manages the overall model flow
        - Space class that defines the space setting and store cell-level information
        - Visualization class that makes informed plots (both static and interactive) (do show the plot!)
        - Save the plot to a local folder called "output_plots"
        - A main function to run the model
        Do NOT provide any explanations or notes outside the code. Just provide the code.

        Before finalizing, ask yourself:
        (1) Are all variables that need to be updated actually updated at each timestep?
        Check every attribute defined in the mechanistic model — does it appear in step()?
        (2) Are there any missing components that are essential for the model to run, such as agent activation, data collection, or visualization?
        (3) Is the simulation schedule implemented in the correct order?
        (4) Is every output variable needed for analysis collected in the DataCollector?
        (5) Are agent interactions implemented correctly (both locally and globally)?
        If you cannot answer any of these questions, the code is likely incomplete or incorrect — revise.

        Include a if __name__ == "__main__": block to run the model for a few steps, print key outputs, an
        An example can be:
        if __name__ == "__main__":
            model = SocialModel(num_agents=100, width=10, height=10)
            for i in range(5):
                model.step()
        """

        user_prompt = f"""
        Please write up the code based on this conceptual model: {conceptual_model}
        Make sure to follow the instructions in the system prompt.
        Taking into account the user requirements: {user_requirements}.
        """

        # call the LLM
        LLM_model = self.client.chat.completions.create(
            model=self.model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # parse the responses
        code = LLM_model.choices[0].message.content
        code = code.replace("```python\n", "").strip() # remove unnecessary markdown formatting if any
        code = code.replace("```", "").strip()
        with open("generated_model.py", "w") as f: # export to a python file
            f.write(code)

        return code
    
    def code_debugging(self, code:str):
        """
        This function will first run the generated code and identify any errors or issues.
        It will return an error message to the code revision_module if any issues are found.
        The input should be a python file?
        """
        # write the code as a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # run the code in an isolated environment for testing
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0: # error spotted, 0 for no error
            print("Error detected during code execution:")
            print(stderr)
            return stderr  # return the error message
        else:
            print("No error detected")
            print(stdout)
            return None # no error message in this case

    def code_revision(self, code:str, error_message:str):
        """
        This function will revise the code based on the provided error message.
        """
        # prompting the LLM
        system_prompt = """
        You are a coding assistant specialized in revising Python code for MESA-based agent-based models.
        Your task is to revise the provided code based on the given error message.
        First analyze the error message carefully to understand what went wrong in the code execution.
        Then identify the specific part(s) of the code that likely caused the error.
        Revise the code to fix the error, ensuring that the revised code is clean, well-commented, modular and runnable.
        Do NOT provide any explanations or notes outside the code. Just provide the revised code.
        """
        user_prompt = f"""
        Please revise the following code based on this error message: {error_message}.
        Here is the original code:
        {code}
        """

        # call the LLM
        LLM_model = self.client.chat.completions.create(
            model=self.model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # parse the responses
        revised_code = LLM_model.choices[0].message.content
        revised_code = revised_code.replace("```python\n", "").strip() # remove unnecessary markdown
        revised_code = revised_code.replace("```", "").strip()

        return revised_code
    
    def model_interface_generation(self, code:str):
        """
        This function will extract a model interface from the generated code.
        The interface will be used later for evaluation and validation.
        """
        # read python code
        with open("generated_model.py", "r") as f:
            code = f.read()
        
        # prompting the LLM
        system_prompt = """
        You are an expert in python programming and software design.
        Your task is to extract a model interface from the provided python code.
        This interface file should contain the following components:
        1. model class name, avaliable functions, implemented framework, and scheduler type.
        2. agent class names, available function, their key attributes (include name, type and range) and decision rules.
        3. environment settings and key parameters (include name, type and range).
        4. decision variables and formula.
        5. output variables and their calculation methods.
        6. data collection methods.
        For each function, specify its name, input parameters (with types), output (with type), and a brief description of its purpose.
        DO NOT invent any new names or parameters yourself. Stick strictly to the names and information from the provided code.
        The output should be a json file with clear structure.
        Do NOT provide any explanations or notes outside the code. Just provide the json interface.
        Return ONLY the JSON array. No markdown, no ```json fences, no commentary.
        """

        user_prompt = f"""
        Please extract a model interface from the following python code: {code}.
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
        )

        # parse the responses
        model_interface = LLM_model.choices[0].message.content
        try:
            interface = json.loads(model_interface)
            return json.dumps(interface, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return model_interface

    def run_pipeline(self,model_save_path:str,code_output_path:str, user_requirements:str):
        """
        This function runs the entire code generation pipeline."""
        print("Step 1: Generating initial code based on the conceptual model...")
        code = self.code_generation(model_save_path, user_requirements)

        print("Step 2: Debugging the generated code...")
        error_message = self.code_debugging(code)
        
        iteration = 1
        while error_message:
            print(f"Error detected in iteration {iteration}:")
            print(error_message)
            print("Step 3: Revising the code based on the error message...")
            code = self.code_revision(code, error_message)
            
            print("Re-debugging the revised code...")
            error_message = self.code_debugging(code)
            iteration += 1

            if iteration > 5:  # limit the number of revision iterations to prevent infinite loops
                print("Seems the LLM is trapped to a deadlock, needs manual revision of the code.")
                break
        
        print("Step 4: Exporting code file...")
        with open(code_output_path, "w") as f: # export to a python file
            f.write(code)
        print("Code generation pipeline completed successfully.")

        print("Step 5: Generating model interface...")
        model_interface = self.model_interface_generation(code)
        interface_output_path = code_output_path.replace(".py", "_interface.json")
        with open(interface_output_path, "w") as f:
            f.write(model_interface)
        print("Model interface generated successfully.")




 

