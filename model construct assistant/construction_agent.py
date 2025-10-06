from openai import OpenAI
import json
import os


class ModelConstructor:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # set your API key in environment variable
        self.model_name = model_name  # default model

    def extract_problem_definition(self, file_path: str):
        """
        This llm agent reads the problem definition provided by users and extracts key information in json format
        """
        # load problem definition given by users
        with open(file_path, "r", encoding="utf-8") as file:
            problem_definition = file.read()

        # prompt the LLM to extract key information
        system_prompt = """
        You are a research assistant who specializes in social simulation models.
        You are given a problem definition provided by users.
        Your task is to extract key information from the problem definition, including:
        1. types and roles of involved agents
        2. environment and interaction structure
        3. key behaviors and decision-making processes of agents
        Output strictly in JSON format:
        {
          "agent_types": ["type1", "type2", "..."],
          "environment": "description of the environment",
          "interaction_structure": "description of interaction structure",
          "key_behaviors": ["behavior1", "behavior2", "..."],
          "decision_making_processes": ["process1", "process2", "..."],
        }
        Ensure your response is valid JSON that can be parsed by a JSON parser.
        """
        user_prompt = problem_definition

        # call the LLM model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        # parse the response to extract JSON
        pb = response.choices[0].message.content
        try:
            features = json.loads(pb)
        except json.JSONDecodeError:
            raise ValueError(
                "The response is not valid JSON. Please check the output format."
            )
        return features

    def generate_agent_rules(self, features: str):
        """
        This llm agents generates agent rules based on the provided context.
        """
        # prompt the llm to generate agent rules
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
        please generate agent rules based on the problem definition {json.dumps(features, indent = 2)}.
        Think step-by-step and determine the agent's situation, and what steps the agent is going to take next.
        Translate this reasoning into a small set of explicit if–then rules.
        """

        # call the LLM model
        llm_model = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # parse the json response
        response = llm_model.choices[0].message.content
        try:
            rules = json.loads(response)
            return json.dumps(rules, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return response

    def model_construction_pipeline(self, file_path: str):
        """
        This function provides a pipeline that:
        (1) extract problem definition from user input;
        (2) generate agent rules based on problem definition;
        """
        # first get the features
        print("Step 1: Extracting problem definition from user input...")
        features = self.extract_problem_definition(file_path)
        print("Extracted features:")
        print(json.dumps(features, indent=2))

        # generate the agent rules
        print("Step 2: Generating agent rules based on problem definition...")
        agent_rules = self.generate_agent_rules(features)
        print("Generated agent rules:")
        print(agent_rules)

    # TODO: save the output into ODD format
