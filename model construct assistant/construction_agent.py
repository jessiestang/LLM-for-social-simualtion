from openai import OpenAI
from docx import Document
import json
import os
import re


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
          ""key parameters": ["parameter1", "parameter2", "..."]
        }
        Ensure your response is valid JSON that can be parsed by a JSON parser.

        Example
        The given problem definition is:
        "We want to model the voting behaviour of citizens in a democratic society.
        Suppose there are two types of voters with different political preferences (+1 and -1).
        Voters interact in a social network where they can influence each other's opinions.
        Each voter decides whether to keep or change their opinin based on that of their neighbours."

        The extracted key information should be:
        {
          "agent_types": ["type1: preference +1", "type2: preference -1"],
          "environment": "Voters are placed in a social network",
          "interaction_structure (among agents)": "voters can read opinions from their neighbours and influence each other",
          "interaction_structure (with environment)": "N/A",
          "key_behaviors": ["interacting with neighbours", "updating own opinion"],
          "decision_making_processes": ["deciding whether to keep or change opinion based on neighbours' opinions"],
          "key parameters": ["network structure", "initial opinion distribution", "threshold for opinion change"]
        }
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
        Your task is to analyze the context and help the researcher brainstorm potential mechanism and agent rules behind the problem context.
        You should perform the following steps:
        (1) Understand the agent's current situation based on the provided context.
        (2) Brainstorm three different kinds of model to simulate the situation.
        (3) For each model, identify:
            (a) At least three relevant social science theories that could inform the model design.
            (b) At least three key mechanisms that drive agent behavior in the model.
            (c) The specific rules that agents would follow for each mechanism, in response to different conditions.

        You should think step-by-step and ensure that the rules are clear, concise, and logically sound.
        The specific rules should and model descriptions should be different for different models.
        The output should be in JSON format and in academic report style
        Output strictly in this json output, specific each model with a title:
        [{ 
        "model 1 title": "A concise title for the first model",
        "problem context": summarize the problem context 2 sentences,
        "model description": "A brief description of you proposed computational model in 2 sentences.",
        "social theories and agent rules": "Give at least three social science theories and explain how they relate to problem context, in 3 sentences.",
        "action rules": "Describe the agent action rules in detail, in 3-5 sentences."},


        {"model 2 title": "A concise title for the second model",
        "problem context": ...,
        "model description": ...,
        ...},

        {"model 3 title": "A concise title for the third model",
        "problem context": ...,
        "model description": ...,
        ...}]
        
        """

        user_prompt = f"""
        please generate agent rules based on the problem definition {json.dumps(features, indent = 2)}.
        Think step-by-step and determine the agent's situation, and what steps the agent is going to take next.
        Think about three different kinds of models to simulate the situation.
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
            if isinstance(rules, dict):
                rules = [rules]  # Ensure it's a list of models
            elif isinstance(rules, list):
                pass
            else:
                raise ValueError("The response is not a valid list or dictionary.")
    
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                rules = json.loads(match.group())
                if isinstance(rules, dict):
                    rules = [rules]  # Ensure it's a list of models
                elif isinstance(rules, list):
                    pass
            else:
                raise ValueError("LLM output not valid JSON, here’s raw output:")
        return rules

    def refine_with_feedback(self, agent_rules: str, user_feedback: str):
        """
        This LLM agent reflects on the generated model rules and suggests improvements. It also takes users's feedback into account
        """
        # load user input
        system_prompt = """You are a research assistant who specializes in social simulation models. You are given a set of model rules and user feedback.
        Your task is to refine the model rules based on the feedback and your own reflection. Output the refined model rules in JSON format."""
        user_prompt = f"""The given model rules are: {agent_rules}. The user feedback is: {user_feedback}. 
        Please refine the model rules based on the feedback and your own reflection. Output the refined model rules in JSON format."""

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
            refined_rules = json.loads(response)
            return json.dumps(refined_rules, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return response

    def model_construction_pipeline(self, file_path: str, save_path: str):
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
        print(json.dumps(agent_rules, indent=2, ensure_ascii=False))

        # ask for user feedback and refine the model
        model_type = int(input("which model do you want to choose (1/2/3)?"))
        selected_rules = agent_rules[model_type - 1]
        selected_model = json.dumps(selected_rules, ensure_ascii=False, indent=2)   # select the chosen model

        user_input = input("do you want to provide any feedback (y/n)?")
        refined_rules = None
        while user_input.lower() == "y":
            user_feedback = input("please provide your feedback:")
            print("Step 3: Refining model based on user feedback...")
            refined_rules = self.refine_with_feedback(selected_model, user_feedback)
            print("Refined agent rules:")
            print(refined_rules)
            user_input = input("do you want to provide any feedback (y/n)?")

        # save and export the conceptual model
        print("Step 4: Saving and exporting the conceptual model...")
        if refined_rules:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(refined_rules, f, indent=2, ensure_ascii=False)  # save json output for code assistant
            self.save_to_wordfile(refined_rules, save_path.replace(".json", ".docx"))  # save word file for users
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(selected_model, f, indent=2, ensure_ascii=False)  # save json output for code assistant
            self.save_to_wordfile(selected_model, save_path.replace(".json", ".docx"))  # save word file for users

    def save_to_wordfile(self, model_description: str, file_path: str):
        """
        This function saves the model description and export it as a word file
        """
        doc = Document()
        doc.add_heading("Conceptual Model Description", level=1)

        # parse the model description
        sections = json.loads(model_description)
        for section, content in sections.items():
            doc.add_heading(section.replace("_", " ").title(), level=2)
            doc.add_paragraph(content)

        doc.save(file_path)  # export to word file

        print(f"Model description saved to {file_path}")

    #TODO: export in ODD format

