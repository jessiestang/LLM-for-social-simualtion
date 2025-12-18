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
    
    def _safe_json_load(self, text: str):
        """
        Robustly try to parse JSON from LLM output.
        Strategies:
        1) Direct `json.loads`.
        2) Extract the first {...} or [...] block and parse.
        3) Escape backslashes in the extracted block and parse (helps with LaTeX like `\beta`).
        Raises ValueError with the raw text when parsing ultimately fails.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract first JSON object/array block
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try escaping single backslashes which commonly break JSON when LLM outputs LaTeX
                escaped = candidate.replace("\\", "\\\\")
                try:
                    return json.loads(escaped)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"LLM output not valid JSON. Raw output:\n{text}")
    
    def generate_decision_rule(self, problem_context:str):
        """
        This LLM agent will identify variables that influence the agent's behavior.
        It will then shows how these variables affect agent's decision-making process together.
        """
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        # system prompt
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Your task is to formalize the decision-making process of agents within a predefined simulation setup
        and express it as both mathematical equations and if–then rules.

        You will be provided with:
        - A description of the model context (e.g., the environment, agent types, and behavioral goal),
        - The decision agents need to make,
        - The relevant variables and parameters.

        Your goal is to:
        (1) Identify the variables that influence the agent’s decision,
        (2) Describe how these variables interact to determine the agent’s perceived state,
        (3) Express this relationship as an explicit formula,
        (4) Translate that formula into clear if–then decision rules,
        (5) Explain the behavioral or social reasoning behind each step.

        Step 1: Variable Identification
        List all variables that influence the decision. For each variable, specify:
        - Name and meaning,
        - Data type (e.g., Boolean, float, integer),
        - How it is updated over time (update rule).

        Step 2: Mechanistic Integration
        Formulate a mathematical equation that combines these variables into a single “decision signal” variable (e.g., perceived support, payoff, or utility).
        - Use proper mathematical notation (e.g., \( O_i^t = β S_i^l + (1−β) S_i^m \)).
        - Define each symbol clearly.
        - If parameters exist (e.g., β, α, θ), describe their range and role.

        When constructing or updating equations, consider not only additive (linear) relationships 
        but also multiplicative, interaction, and nonlinear effects where theoretically justified.

        - Interaction terms (e.g., X * Y) can represent how one factor amplifies or moderates another.
        - Nonlinear transformations (e.g., logistic, exponential, or squared terms) can represent thresholds or saturation effects.
        - Temporal feedback (e.g., variable depends on its own past value) can represent learning or adaptation.

        Prefer interpretable complexity: always explain what social mechanism each nonlinearity represents.

        Step 3: Decision Rule Construction
        Translate the equation into one or more explicit if–then statements.
        Example:
        IF perceived_majority > threshold THEN speak ELSE stay_silent.

        Step 4: Reasoning
        For each rule, explain briefly why this rule makes sense given the model context.

        Output requirements:
        - Output *only valid JSON* (no markdown, no text before or after).
        - Use the following JSON schema strictly:
        {
        "decision_context": "what the agent is deciding",
        "variables": [
            {
            "name": "variable_name",
            "meaning": "what it represents conceptually",
            "data_type": "data type (e.g., float, integer, boolean)",
            "update_rule": "how it changes over time"
            }
        ],
        "formula": {
            "expression": "mathematical formula (use LaTeX notation if needed)",
            "definitions": {
            "symbol": "what it means and its range"
            }
        },
        "decision_rules": [
            {
            "rule": "IF condition THEN action",
            "explanation": "reasoning behind the rule"
            }
        ],
        "parameters": {
            "parameter_name": "description of its role and expected range"
        },
        "assumptions": [
            "state any simplifying assumptions or constraints"
        ]
        }"""

        # user prompt
        user_prompt = f"""
        The provided research context is {problem_definition}. Stick strictly to the requirement in the system prompt."""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        pb = response.choices[0].message.content
        features = self._safe_json_load(pb)
        return features

    def human_in_the_loop(self, agent_rules: str, user_feedback: str):
        """
        Users can provide new ideas of potential variables they think will also be useful in explaining agents'decisions.
        This LLM agent will try to see how it can also merge the proposed variable into existed design.
        """
        # load user input
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Your task is to revise and extend an existing agent decision-making formulation,
        based on new variables suggested by human researchers.

        You will be provided with:
        (1) A preliminary set of variables that affect agent behavior, along with their mathematical relationships.
        (2) A list of additional variables proposed by human researchers.

        Your goals are to:
        1. Assign an appropriate data type, value range, and threshold logic to each newly proposed variable.
        2. Analyze how these new variables might interact with the existing ones.
        3. Integrate them into a new or revised mathematical formula that represents agent decision-making.
        4. Translate the updated formula into explicit if–then decision rules.
        5. Provide a short behavioral or social explanation for each rule.

      
        Step 1: Variable Definition
        For each variable (both existing and new):
        - Specify its name, conceptual meaning, data type (e.g., float, integer, Boolean),
        - Its plausible range or domain,
        - The logic that determines its threshold or triggering value.

        Step 2: Relationship Formulation
        Explain how the newly proposed variables influence the agent’s perception or decision-making process.

        Step 3: Equation Construction
        Write an updated mathematical equation that captures the relationships among all variables.
        Use *LaTeX notation* for readability (e.g., \( O_i^t = βS_i^l + (1−β)S_i^m + γC_i \)).
        Define each symbol precisely, including any new parameters introduced.

        Step 4: Decision Rule Derivation
        Translate the new or revised formula into one or more *if–then decision rules*

        Step 5: Reasoning
        For each rule, explain the behavioral or psychological mechanism that motivates it.

        Output Requirements:
        Return *only valid JSON* (no text outside the JSON object). Use this schema strictly:

        {
        "updated_variables": [
            {
            "name": "variable_name",
            "meaning": "what it represents conceptually",
            "data_type": "float / integer / boolean",
            "range": "expected numerical range or domain",
            "threshold_logic": "how it affects agent decision-making"
            }
        ],
        "new_relationships": [
            "short natural-language description of how new variables interact with existing ones"
        ],
        "updated_formula": {
            "expression": "mathematical formula (in LaTeX notation)",
            "definitions": {
            "symbol": "what it means and its range"
            }
        },
        "decision_rules": [
            {
            "rule": "IF condition THEN action",
            "explanation": "why this rule makes behavioral sense"
            }
        ],
        "assumptions": [
            "state any simplifying assumptions or constraints introduced"
        ]
        }
        """
        user_prompt = f"""The previous design is: {agent_rules}. The user input: {user_feedback}. Stick strictly to the requirement in the system prompt.
        """

        # call the LLM model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        pb = response.choices[0].message.content
        features = self._safe_json_load(pb)
        return features
    
    def export_descriptive_model(self, problem_context:str, preliminary_model:str, secondary_model:str = None):
        """
        This LLM agent will regorganize and export a complete and descriptive mechanistic model
        based on the problem context, the preliminary decision-making part, and the secondar decision-making part,
        if available.
        """
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        system_prompt="""
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Your task is to turn a few notes of a model design into a complete, descriptive and formal machanistic model design.
        
        """


    
    def model_brainstorming(self, problem_context: str):
        """
        This llm agent will do three tasks:
        (1) reconstruct the classic agent-based model based on the problem context provided by users;
        (2) propose alternatives or novel variants of the classic model;
        (3) validate suggestions by literature (to be implemented in future).
        """
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        # prompt the LLM to brainstorm the actions of agents, and the motivations behind the actions
        system_prompt = """
        You are a research assistant in computational social science and agent-based modeling.
        You specialize in brainstorming how classic agent-based models can be extended to capture new social mechanisms.
        You will be provided with a problem context.
        You have three tasks. Please accomplish them step-by-step:
        (1) Reconstruct the traditional agent-based model that best fits the problem context.
        (2) Propose at least three novel extensions of the traditional model to better capture the social mechanisms in the problem context.
        (3) For each proposed extension, provide literature references that support your suggestions.

        For each version of model, please describe:
        (1) key agents and their attributes;
        (2) environment and interaction structure;
        (3) expected emergent behaviors;
        (4) theoretical justifications of extended rules from literature.
        (5) why this new variant can yield new insights

        Firstly, determine if user has provided any information in the problem context, regarding the 5 aspects mentioned above.
        If they provide relevant information, please make sure to incorporate them into your model design (for both traditional and extended models).
        If not, you can make reasonable assumptions based on your internal reasoning and external literature knowledge.
        provide your reseasoning and assumptions clearly in your output. Examples of such reasoning could be:
        "The problem context specifies the model structure: a lattice grid where agents interact with their immediate neighbors.
        However, it does not define what types of neighbours will be used. I will suggest two types of neighbour structures: von Neumann and Moore neighbourhoods,
        as they are commonly used in agent-based modeling literature"
        "The problem context mentions that agents can have two actions: move or stay. I will incorporate this information into the agent action design."
        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.

        Output strictly in JSON format:
        [{
            "model_version": "traditional" or "extension_1" or "extension_2",
            "key_agents": "description of key agents",
            "agent_attributes": "description of agent attributes",
            "agent_actions": "description of actions agent can perform",
            "environment": "description of the environment structure",
            "interaction_structure": "description of interaction structure among agents and with environment",
            "expected_emergent_behaviors": "description of expected emergent behaviors",
            "theoretical_justifications": "use theories from literature to justify the extended rules",
            "insightfulness": "description of why this variant can yield new insights" #  extension only
            "external_system": "if the user suggests the operation of an external system to the environment and agent"
          },
        
          {"model_version": "...",
          "key_agents": "...",
          "agent_attributes": "...",
          ...
          },

          {"model_version": "...",
          "key_agents": "...",
          "agent_attributes": "...",
          ...
          }]
        """
        user_prompt = f"""The given problem context is:
        {problem_definition}
        Please brainstorm the model construction based on the three tasks mentioned above.
        """
        # call the LLM model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        # parse the response to extract JSON
        response = response.choices[0].message.content
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

    
    
    def model_generation_refining(self, features: str):
        """
        This llm agents refine the brainstorming ideas and transform it into concrete description of the model
        """
        # prompt the llm to generate agent rules
        system_prompt = """
        You are a helpful research assistant who specializes in constructing social simulation models based on provided context.
        You will be provided with a short context of an agent-based model.
        Your task is to dive deeper into the context and enlarge the context into a specific conceptual model for agent-based modeling.
        For baseline model, stick to the provided context strictly and do not add any extra assumptions.
        For extended models, you can make reasonable extensions based on your internal reasoning and external literature knowledge.

        You should follow these steps:
        (1) Think about how the agents will be initialized. What attributes will they have? What actions can they perform?
        (2) Consider the environment in which the agents operate. How is it structured? How do agents interact with each other and with the environment?
        (3) Define the decision-making processes of the agents. How do they decide what actions to take based on their attributes and the state of the environment?
        You should define the decision-making processes strictly based on what you have in the agent initialization and environment setup.
        (4) Specify the rules that govern agent behavior. What conditions lead to specific actions? How do agents adapt or learn over time?
        You should define the behavior rules strictly based on what you have in the agent initialization, environment setup, and decision-making processes.
        (5) Finally, outline the expected emergent behaviors that arise from the interactions of agents within the environment.
        This should be a direct consequence of the agent rules you have defined, plus a little bit reasoning and analysis.

        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.
        Start your response directly with { and end with }.

        Your output should be in JSON format, clearly outlining the agent rules and model structure:
        {
            "agent_initialization": {
                "attributes": ["attribute1", "attribute2", "attribute3"], # stick to this number for now
                "actions": ["action1", "action2", "actin3"], # stick to this number for now
                "agent_types": ["type1", "type2"], # stick to this number for now
                "initial_distribution": "description of how agents are initially distributed",
                "size of agents": "number of agents in the model",
                "adaptation_mechanisms": "description of how agents adapt or learn over time"
            },
            "environment": {
                "structure": "description of the environment structure",
                "interactions": "description of agent-agent and agent-environment interactions",
                "changes_over_time": "description of how the environment changes over time, and how agents influence these changes",
                "key_parameters_that_controls_the_environment": ["parameter1", "parameter2"], # stick to this number for now
            },
            "decision_making_processes": ["process1", "process2", "..."],
            "behavior_rules": [
                "Rule 1: specific behavior description",
                "Rule 2: specific behavior description",
                "Rule 3: specific behavior description",
            ], # stick to this number for now
            "expected_emergent_behaviors": [
                "Behavior 1: what emerges",
                "Behavior 2: what emerges",
                "..."
            ]
            }
        Ensure your response is valid JSON that can be parsed by a JSON parser.
        """

        user_prompt = f"""
        please construct a very detailed agent-based model based on this brainstorming idea {json.dumps(features, indent = 2)}.
        Follow the steps mentioned in the system prompt to generate a comprehensive model description with clear agent rules.
        Output ONLY the JSON object, no other text.
        Please strictly stick to the provided context and do not add any extra assumptions!!
        """

        # call the LLM model
        llm_model = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
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


    def model_construction_pipeline(self, file_path: str, save_path: str):
        """
        This function provides a pipeline that:
        (1) brainstorming model ideas based on problem definition;
        (2) generate conceptual model based on brainstorming ideas;
        (3) refine the model based on user feedback;
        (4) save and export the conceptual model.
        """
        # first get the features
        print("Step 1: Brainstorming model ideas based on problem definition...")
        model_ideas = self.model_brainstorming(file_path)
        print("Brainstormed model ideas:")
        print(json.dumps(model_ideas, indent=2))
        idea_type = int(input("which model do you want to choose (1/2/3)?"))
        selected_idea = model_ideas[idea_type - 1]
        #TODO: enable human feedback here to regenerate ideas if needed

        # generate the agent rules
        print("Step 2: Generating the conceptual model based on the selected idea")
        conceptual_model = self.model_generation_refining(selected_idea)
        print("Generated conceptual model:")
        print(json.dumps(conceptual_model, indent=2, ensure_ascii=False))

        user_input = input("do you want to provide any feedback (y/n)?")
        refined_model = None
        while user_input.lower() == "y":
            user_feedback = input("please provide your feedback:")
            print("Step 3: Refining model based on user feedback...")
            refined_model = self.refine_with_feedback(conceptual_model, user_feedback)
            print("Refined conceptual model:")
            print(refined_model)
            user_input = input("do you want to provide any feedback (y/n)?")

        # save and export the conceptual model
        print("Step 4: Saving and exporting the conceptual model...")
        if refined_model:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(refined_model, f, indent=2, ensure_ascii=False)  # save json output for code assistant
            #self.save_to_wordfile(refined_model, save_path.replace(".json", ".docx"))
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(conceptual_model, f, indent=2, ensure_ascii=False)  # save json output for code assistant
            #self.save_to_wordfile(conceptual_model, save_path.replace(".json", ".docx"))  # save word file for users
    
    def new_pipeline(self,file_path: str):
        print("Step 1: Brainstorming model ideas based on problem definition...")
        mechanistic_model = self.generate_decision_rule(file_path)
        print(json.dumps(mechanistic_model, indent=2))
        flag = input("Do you want to propose any new variable based on the current design? (y/n)")
        while flag == "y":
            user_input = input("Please provide name and a short definition of each new variable")
            print("Step21: Generating new model ideas based on your input")
            new_model = self.human_in_the_loop(json.dumps(mechanistic_model, indent=2), user_input)
            print(json.dumps(new_model, indent=2))
            flag = input("Do you want to propose any new variable based on the current design? (y/n)")


        

    def save_to_wordfile(self, model_description: str, file_path: str):
        """
        This function saves the model description and export it as a word file
        """
        doc = Document()
        doc.add_heading("Conceptual Model Description", level=1)

        # Handle both string and dict/list inputs
        if isinstance(model_description, str):
            sections = json.loads(model_description)
        elif isinstance(model_description, (dict, list)):
            sections = model_description
        else:
            raise ValueError(f"Unexpected type for model_description: {type(model_description)}")
        
            # Handle list of models
        if isinstance(sections, list):
            for i, model in enumerate(sections, 1):
                doc.add_heading(f"Model {i}", level=1)
                for section, content in model.items():
                    doc.add_heading(section.replace("_", " ").title(), level=2)
                    
                    # Handle nested structures
                    if isinstance(content, dict):
                        for key, value in content.items():
                            doc.add_paragraph(f"{key.replace('_', ' ').title()}: {value}")
                    elif isinstance(content, list):
                        for item in content:
                            doc.add_paragraph(str(item), style='List Bullet')
                    else:
                        doc.add_paragraph(str(content))
                
                if i < len(sections):
                    doc.add_page_break()
        
        # Handle single model (dict)
        else:
            for section, content in sections.items():
                doc.add_heading(section.replace("_", " ").title(), level=2)
                
                # Handle nested structures
                if isinstance(content, dict):
                    for key, value in content.items():
                        doc.add_paragraph(f"{key.replace('_', ' ').title()}: {value}")
                elif isinstance(content, list):
                    for item in content:
                        doc.add_paragraph(str(item), style='List Bullet')
                else:
                    doc.add_paragraph(str(content))

        doc.save(file_path)
        print(f"Model description saved to {file_path}")

    #TODO: export in ODD format

