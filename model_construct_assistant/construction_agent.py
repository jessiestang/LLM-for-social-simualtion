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
        - How it is updated over time (update rule and temporal scope: per time step, cumulative, or adaptive).

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

        Step 5: Identify externl systems that interact with agents, if available.
        You can SKIP this step if no external system is identified, also in the output.
        Only execute step 5 if it is clearly mentioned in the input file that there is an external system.
        For each system:
        - Specify what input it receives from agents;
        - Specify how it processes that input;
        - Specify what output it returns to agents;
        - Determine the timing of interaction (before or after agent updates)

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
        ],
        "external_systems": [
        {
            "system_name": "name of the external system",
            "inputs_from_agents": "what data or messages the system receives from agents",
            "internal_variables": [
            {
                "name": "variable_name",
                "meaning": "what it represents inside the system",
                "update_rule": "how it updates over time (mathematical formula or aggregation rule)"
            }
            ],
            "processing_logic": [
            {
                "rule": "explicit mathematical or algorithmic rule",
                "description": "what this rule does and why it matters"
            }
            ],
            "outputs_to_agents": [
            {
                "output_variable": "name of what is sent back to agents",
                "generation_rule": "how it is computed from the system’s internal state",
                "timing": "when it becomes available to agents (before/after their next decision)"
            }
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
        Your task is to transform short, partially specified model notes into a complete,
        descriptive, and formal mechanistic model design suitable for implementation and analysis.

        You will be provided with:
        (1) A problem context file describing the research question, agent attributes, and environmental setup;
        (2) A JSON file describing the agent decision context, behavioral rules, variables, thresholds, and model parameters.

        Your goals are to:
        - Produce a complete and structured model specification containing all necessary components for implementation;
        - Include mathematical formulations of agent decision-making, using appropriate notation (LaTeX-style or inline);
        - Provide a natural-language behavioral explanation corresponding to each equation or rule;
        - Ensure that no new information or assumptions are introduced beyond what appears in the provided files.

        Follow these steps carefully:

        1. Identify all agent types, their attributes, and actions implied by the rules.
        2. Define the environment (e.g., grid, network) and its parameters.
        3. For each behavioral rule, provide both:
        - the mathematical expression governing it,
        - a verbal explanation in plain academic English,
        - description of how variables in the mathematical expression are updated.
        4. Summarize model-level mechanisms such as feedback loops or external influences.
        5. List all simulation parameters, including agent population size, time steps, constants, and sensitivity parameters.
        6. Provide a short academic-style description (4-5 sentences) summarizing the full model design.

        Output strictly as *valid JSON* (no Markdown, no commentary).
        Use the following schema exactly:

        {
        "model_title": "short descriptive title",
        "overview": "brief purpose of the model",
        "agents": {
            "types": ["AgentType1", "AgentType2"],
            "attributes": {
            "AgentType1": ["attr1", "attr2"],
            "AgentType2": ["attr1", "attr2"]
            },
            "actions": {
            "AgentType1": ["action1", "action2"],
            "AgentType2": ["action1", "action2"]
            }
        },
        "environment": {
            "structure": "description of environment (e.g., grid, network)",
            "interaction_rules": "how agents interact with neighbours or the environment",
            "parameters": {
            "param1": "description of parameter 1",
            "param2": "description of parameter 2"
            }
        },
        "external_systems": [
        {
            "system_name": "name of the external system",
            "inputs_from_agents": "what data or messages the system receives from agents",
            "internal_variables": [
            {
                "name": "variable_name",
                "meaning": "what it represents inside the system",
                "update_rule": "how it updates over time (mathematical formula or aggregation rule)"
            }
            ],
            "processing_logic": [
            {
                "rule": "explicit mathematical or algorithmic rule",
                "description": "what this rule does and why it matters"
            }
            ],
            "outputs_to_agents": [
            {
                "output_variable": "name of what is sent back to agents",
                "generation_rule": "how it is computed from the system’s internal state",
                "timing": "when it becomes available to agents (before/after their next decision)"
            }
            ],
  
        "decision_logic": {
            "mathematical_expressions": [
            {
                "equation": "y_i(t) = β * x_i(t) + (1 - β) * m_i(t)",
                "description": "brief explanation of what this equation means",
                "variables":{
                "variable_name_1": "descrpition of how the value of this variable can be obtained and updated",
                "variable_name_2": "descrpition of how the value of this variable can be obtained and updated"
                }
            }
            ],
            "if_then_rules": [
            {
                "rule": "IF condition THEN action",
                "explanation": "social or behavioral reasoning behind the rule"
            }
            ]
        },
        "simulation_parameters": {
            "num_agents": "number of agents in the model",
            "time_steps": "number of iterations to simulate",
            "constants": {
            "constant_name_1": "description of what this constant does",
            "constant_name_2": "description of what this constant does"
            }
        },
        "description_of_the_model": "A concise, academic summary (4-5 sentences) describing how the model operates and what emergent phenomena it captures."
        }
        """
        if secondary_model:
            user_prompt = f"""Here is the input problem context: {problem_definition},
            and here are the JSON files describing the agents'behavior and decision-making {preliminary_model}, {secondary_model}.
            Stick strictly to the instructions from the system prompt"""
        else:
            user_prompt = f"""Here is the input problem context: {problem_definition},
            and here is the JSON file describing the agents'behavior and decision-making {preliminary_model}.
            Stick strictly to the instructions from the system prompt"""
        
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

    def new_pipeline(self,file_path: str, save_path:str):
        print("Step 1: Brainstorming model ideas based on problem definition...")
        mechanistic_model = self.generate_decision_rule(file_path)
        print(json.dumps(mechanistic_model, indent=2))

        new_model = None
        flag = input("Do you want to propose any new variable based on the current design? (y/n)")
        while flag == "y":
            user_input = input("Please provide name and a short definition of each new variable")
            print("Step21: Generating new model ideas based on your input")
            new_model = self.human_in_the_loop(json.dumps(mechanistic_model, indent=2), user_input)
            print(json.dumps(new_model, indent=2))
            flag = input("Do you want to propose any new variable based on the current design? (y/n)")
        
        print("Step 3: Summarizing the modelling ideas...")
        if new_model:
            summary = self.export_descriptive_model(file_path, mechanistic_model, new_model)
            print(summary)
        else:
            summary = self.export_descriptive_model(file_path, mechanistic_model)
            print(summary)
        
        print("Step 4: Export the model...")
        if summary:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
         

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

