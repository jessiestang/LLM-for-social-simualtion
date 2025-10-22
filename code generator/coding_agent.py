# implement the coding agent
import os
from openai import OpenAI
import json
import re
import traceback

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
        You file should contain at least following components:
        - Agent class with agent types, attributes and methods
        - Model class with scheduler, grid and interaction among agents
        - Environment class with its attributes and methods
        - Visualization class to collect data and make informed visualizations (do show the plot!)
        - A main function to run the model
        Do NOT provide any explanations or notes outside the code. Just provide the code.

        Include a if __name__ == "__main__": block to run the model for a few steps, print key outputs, and show the visualizations for testing.
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
        code = code.replace("```python\n", "").strip() # remove unnecessary markdown formatting if any
        code = code.replace("```", "").strip()
        with open("generated_model.py", "w") as f: # export to a python file
            f.write(code)

        return code
    
    def code_debugging(self, code):
        """
        This function will first run the generated code and identify any errors or issues.
        It will return an error message to the code revision_module if any issues are found.
        """
        try:
            exec(code)
            return None # no error detected
        except Exception as e:
            return traceback.format_exc()

    def code_revision(self, code, error_message):
        """
        This function will revise the code based on the provided error message.
        """
        # prompting the LLM
        system_prompt = """
        You are a coding assistant specialized in revising Python code for MESA-based agent-based models.
        Your task is to revise the provided code based on the given error message.
        Please ensure that the revised code is clean, well-commented, modular and runnable.
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
            temperature=0.2,
            max_tokens=1500
        )

        # parse the responses
        revised_code = LLM_model.choices[0].message.content
        revised_code = revised_code.replace("```python\n", "").strip() # remove unnecessary markdown
        revised_code = revised_code.replace("```", "").strip()

        return revised_code
    
    def run_pipeline(self,json_path):
        """
        This function runs the entire code generation pipeline."""
        print("Step 1: Generating initial code based on the conceptual model...")
        code = self.code_generation(json_path)
        
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
        
        print("Step 4: Exporting code file...")
        with open("generated_model.py", "w") as f: # export to a python file
            f.write(code)
        print("Code generation pipeline completed successfully.")




 #TODO; Still need to figure out LLM Orchestration for master and sub-coding agents!!!
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
        system_prompt = f"""
        You are a coding assistant specialized in generating Python code for MESA-based agent-based models.
        Your task is to generate code for a specific component of the model.
        You will receive shared instructions and a conceptual model as input from the master agent.
        Please stick strictly to the shared_instructions, which state which class name, variable name, function name, parameters to use.
        The shared_instructions also specify which package you can import.
        Do not invent any new names or parameters, to ensure the compatibility with other code files.
        The code file should be clean, well-commented, modular and runnable.
        Do NOT provide any explanations or notes outside the code. Just provide the code for the specified module.

        Only generate the code for the specified module: {self.module_name}, do not write the whole codebase.
        If you are writing:
        - agent.py: define the agent class with its attributes and methods.
        - environment.py: define the environment setting and relevant functions.
        - model.py: define the model class, scheduler, and interaction among agents, and between agents and the environment.
        - run.py: set up the model parameters, initialize the model, and run the simulation loop.
        - visualization.py: collect relevant data and statistics and make informative visualizations.
        Only focus on your relevant module in {shared_instructions}.
        """

        user_prompt = f"""
        Please write up the MESA code for the {self.module_name} component based on the following shared instructions: {shared_instructions}.
        The overall conceptual model is: {conceptual_model}. Please only generate the code for the specified module: {self.module_name}.
        Please stick strictly to {shared_instructions}. Do not invent any new names or parameters.
        """

        # call the LLM
        LLM_model = self.client.chat.completions.create(
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

class MasterAgent:
    def __init__(self, conceptual_model, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # set your API key in environment variable
        self.model_name = model_name  # default model (maybe a stronger model is needed here)
        self.conceptual_model = conceptual_model # input json path here
        self.sub_agents = {
            "agent.py": SubCodeAgents("agent.py", model_name),
            "environment.py": SubCodeAgents("environment.py", model_name),
            "model.py": SubCodeAgents("model.py", model_name),
            "run.py": SubCodeAgents("run.py", model_name),
            "visualization.py": SubCodeAgents("visualization.py", model_name)
        }  # sub-coding agents of the master agent
        self.sub_agents_role = {
            "agent.py": "Responsible for generating the agent class with its attributes and methods.",
            "environment.py": "Responsible for generating the environment setting and relevant functions.",
            "model.py": "Responsible for generating the model class, scheduler, and interaction among agents, and between agents and the environment.",
            "run.py": "Responsible for generating the setup of model parameters, initialization of the model, and running of the simulation.",
            "visualization.py": "Responsible for generating the data collection and visualization functions."
        } # pass on the role description to each sub-coding agent

    """
    The master agent supervises and coordinate the whole code generation process.
    Its task involves:
    (1) Read the input conceptual model and decompose it into sub-components
    (agent.py, environment.py, model.py, run.py, visualization.py).
    (2) Create a list of shared instructions for all sub-coding agents, specifying coding conventions,
    dependencies, and integration points to ensure compatibility across different code files.
    (3) Assign each sum-component and the list of shared insturctions to the sub-coding agents.
    (4) Collect the generated code snippets from each sub-coding agent and integrate them into a cohesive codebase.
    (5) Perform a final review and run the integrated code to ensure functionality and correctness.
    (6) If issues arise, provide feedback to the relevant sub-coding agent for revisions
    (7) Output the final complete codebase if all components function correctly.
    """

    def shared_instructions_generator(self):
        """
        This function generates a list of shared instructions for all sub-coding agents
        based on the overall conceptual model.
        """
        # load the conceptual model
        with open(self.conceptual_model, "r") as f:
            conceptual_model = json.load(f)

        # prompting the LLM (revise later after studying the mesa document)
        system_prompt = """
        You are a coding assistant specialized in social simulation models using MESA framework.
        You task is to generate a list of shared instructions for multiple sub-coding agents.
        These instructions should ensure compatibility and coherence across different code files.
        Your output should be a json list of instructions, following this structure:
        { "agents":[
            {"class_name": ...,
            "attributes": [...],
            "methods": [...]}
            ],
        "model":[{
            "class_name": ...,
            "Scheduler": [...],
            "grid_type": [...],
            "imports": [...]}
            ],
        "parameters":{...},
        "dependencies":{...}
        }
        An example insturction can be:
        { "agents":[
            {"class_name": SocialAgent,
            "attributes": ["opinions", "neighbours"],
            "methods": ["step()", "interact()"]}
            ],
        "model":[{
            "class_name": SocialModel,
            "Scheduler": ["RandomActivation"],
            "grid_type": ["MultiGrid"],
            "imports": ["SocialAgent from agent.py"]}
            ],
        "parameters":{"num_agents": 100, "width": 10, "height": 10},
        "dependencies":{"model.py": ["agent.py", "environment.py"]}
        }
        """

        user_prompt = f"""
        Please generate a list of shared instructions for multiple sub-coding agents based on the following conceptual model: {conceptual_model}.
        The instructions should ensure compatibility and coherence across different code files.
        Please follow the structure and example provided in the system prompt.
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
        try:
            shared_instructions = json.loads(LLM_model.choices[0].message.content)
        except json.JSONDecodeError:
            print("Error decoding JSON from LLM response:")
            match = re.search(r'\{.*\}', LLM_model.choices[0].message.content, re.DOTALL)
            if match:
                shared_instructions = json.loads(match.group(0))
            else:
                print("No valid JSON found in LLM response:")
                raise ValueError("Invalid JSON response from LLM")
            
        return shared_instructions

    def task_division(self,shared_instructions):
        """
        This function divides the overall coding task into sub-components.
        It then assigns each sub-component to a specific sub-coding agent with the shared instructions
        """
        # load the instructions and conceptual model
        # shared_instructions = json.dumps(shared_instructions, indent=2)
        with open(self.conceptual_model, "r") as f:
            conceptual_model = json.load(f)

        code_snippets = {}
        for module_name, sub_agent in self.sub_agents.items(): # assign task to each sub-coding agent
            code = sub_agent.code_generator(shared_instructions, conceptual_model)
            code = code.replace("```python\n", "").strip() # remove unnecessary markdown formatting if any
            code = code.replace("```", "").strip()

            print(f"Generated code for {module_name}")
            code_snippets[module_name] = code # collect the generated code

        return code_snippets

    def code_integration_check(self, code_snippets):
        """
        This function collects all code snippets from sub-coding agents and do a testing run.
        It reports issues and feedbacks to the relevant sub-coding agents for revision.
        It outputs the final complete codebase if all components function correctly.
        """
        failed = []
        for module_name, code in code_snippets.items():
            try:
                compile(code, module_name, 'exec')
                print(f"{module_name} compiled successfully.")
            except SyntaxError as e:
                print(f"Syntax error in {module_name}: {e}")
                failed.append(module_name)
        
        # save the code to a temporary directory
        file_name = "generated_code"
        os.makedirs(file_name, exist_ok=True)
        for module_name, code in code_snippets.items():
            with open(os.path.join(file_name, module_name), "w") as f:
                f.write(code)
        
        return failed

    def run_pipeline(self):
        """
        This function runs the entire code generation pipeline.
        """
        print("Step 1: Generating shared instructions based on the conceptual model...")
        shared_instructions = self.shared_instructions_generator()
        print(f"Shared Instructions: {shared_instructions}")

        print("Step 2: Dividing tasks among sub-coding agents...")
        code_snippets = self.task_division(shared_instructions)
        print(f"Generated key files: {code_snippets.keys()}")

        print("Step 3: Integrating code snippets and performing testing...")
        failed_modules = self.code_integration_check(code_snippets)
        if failed_modules:
            print(f"Code generation pipeline completed with errors in: {failed_modules}")
        else:
            print("Code generation pipeline completed successfully.")

