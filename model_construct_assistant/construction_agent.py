from openai import OpenAI
from docx import Document
import sys
import json
import os
import re


class ModelConstructor:
    def __init__(self, model_name="gpt-4o-mini", logger = None):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # set your API key in environment variable
        self.model_name = model_name  # default model
        self.logger = logger  # initialize the logger
    
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
        List about three variables that influence the decision, based on the provided model context and your internal knowledge. For each variable, specify:
        - Name and meaning,
        - Data type (e.g., Boolean, float, integer),
        - How it is updated over time (update rule and temporal scope: per time step, cumulative, or adaptive).
        - A reasonable default value based on theoretical considerations.

        Step 2: Mechanistic Integration
        Formulate a mathematical equation that combines these agent variables into a single “decision signal” variable (e.g., perceived support, payoff, or utility).
        - Use proper mathematical notation (e.g., \( O_i^t = β S_i^l + (1−β) S_i^m \)).
        - Define each symbol clearly.
        - If parameters exist (e.g., β, α, θ), describe their range and role.
        - If parameters exist, suggest reasonable default values based on theoretical considerations.
        - Provide a short explanation of the behavioral or social mechanism that motivates this equation.

        When constructing or updating equations, consider not only additive (linear) relationships 
        but also multiplicative, interaction, and nonlinear effects where theoretically justified.

        - Interaction terms (e.g., X * Y) can represent how one factor amplifies or moderates another.
        - Nonlinear transformations (e.g., logistic, exponential, or squared terms) can represent thresholds or saturation effects.
        - Temporal feedback (e.g., variable depends on its own past value) can represent learning or adaptation.

        Step 3: Decision Rule Construction
        Translate the equation into one or more explicit if–then statements.
        Example:
        IF [variable] > [threshold] THEN [do action A] ELSE [do action B].

        Step 4: Reasoning
        For each rule, explain briefly why this rule makes sense given the model context.
        
        Step 5: Description temporal ordering
        Provide the full simulation schedule in ordered steps.

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
            "update_rule": "how it changes over time",
            "default_value": "a reasonable default value",
            "range": "expected numerical range or domain"
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
            "parameter_name": "description of its role and expected range",
            "default_value": "suggestion of a reasonable default value based on theoretical considerations"
        },
        "assumptions": [
            "state any simplifying assumptions or constraints"
        ],

        "temporal_ordering": 
            [ "Step 1:...",
            "Step 2:...",
            "Step 3:..."
            ]
        }"""

        # user prompt
        user_prompt = f"""
        The provided research context is {problem_definition}. Stick strictly to the requirement in the system prompt."""
        
        # log the prompt


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

    def decision_rule_variables(self, problem_context:str):
        """
        This LLM agent takes the problem context as input and think about what variables might by important in driveing the agent's baheviour change."""
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Given the problem context, think about what variables or factors might be important in driving the agent's behavior change, if you were the agent in the model.
        Output a list of potential variables that could influence the agent's decision-making process, along with a brief explanation of why each variable might be relevant.
        Output strictly as *valid JSON* with the following schema:
        Do not give more than 5 variables.
        Check if the variables you proposed are logically consistent with each other and with the problem context.
        {
            "potential_variables": [
                {
                    "name": "variable_name",
                    "explanation": "brief explanation of why this variable is relevant, and how it will affect the decision-making process"
                    "data_type": "data type (e.g., float, integer, boolean)",
                    "update_rule": "how it changes over time",
                }
            ]
        }
        """
        user_prompt = f"""The provided research context is {problem_definition}. Stick strictly to the requirement in the system prompt."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        pb = response.choices[0].message.content
        variables = self._safe_json_load(pb)
        return variables
    
    def decision_rule_designer(self, problem_context:str, variables:json):
        """
        This LLM agent will take a list of variables and the problem context as input.
        It will think about how these variables interact together to drive the agent's decision-making process.
        It will express the relationship in if-then rules"""
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        You will be provided with a problem context and a list of variables that are relevant to the agent's decision-making process.
        Your task is to think about how these variables interact together to drive the agent's decision-making process, and express this relationship in if-then rules.
        Select three variables from the provided list that you think are most important in influencing the agent's decision.
        Use these variables then to compile if-then rules that capture the core decision logic of the agent.
        Check if the rules are logically consistent with each other and with the problem context.
        For each behavioural decision described in the problem context, write only one if-then rule that captures the core decision logic of the agent.
        Output strictly as valid JSON with the following schema:
        {   "selected_variables": ["list of the three selected variables that are most important in influencing the agent's decision"],
            "decision_rules": [
                {
                    "rule": "IF condition THEN action",
                    "explanation": "brief explanation of the reasoning behind this rule, and how it relates to the problem context"
                    "mechanistic version": "a mathematical formula that captures the relationship between the variables in this rule (use LaTeX notation if needed)"
                }
            ]
            }
        """

        user_prompt = f"""The provided research context is {problem_definition}, and here are the relevant variables: {json.dumps(variables, indent=2)}.
        Stick strictly to the requirement in the system prompt."""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        pb = response.choices[0].message.content
        decision_rules = self._safe_json_load(pb)
        return decision_rules
    

    def human_in_the_loop(self, agent_rules: str, user_feedback: str):
        """
        Users can provide new ideas of potential variables they think will also be useful in explaining agents'decisions.
        They can also provide feedback on the variables proposed by the LLM, such as modifying the definition, update rule, or default value of the variable.
        This LLM agent will try to see how it can also merge the proposed variable into existed design.
        """
        # load user input
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Your task is to revise and extend an existing agent decision-making formulation, based on new variables suggested by human researchers.

        You will be provided with:
        (1) A preliminary set of variables that affect agent behavior, along with their mathematical relationships.
        (2) A list of additional variables proposed by human researchers, or modifications to the existing variables.

        Your goals are to:
        1. Decide if the proposed changes are new variables or modifications to existing ones.
        2.1 If they are new variables, assign an appropriate data type, value range, and threshold logic to it.
        2.2 If they are modifications, update the variable definition accordingly.
        3. Analyze how these new variables might interact with the existing ones.
        4. Integrate them into a new or revised mathematical formula that represents agent decision-making.
        5. Translate the updated formula into explicit if–then decision rules.
        6. Provide a short behavioral or social explanation for each rule.

      
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
    
    def mechanism_translation(self, problem_context:str, variables:json, decision_rules:json):
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
        (2) A JSON file describing variables that drive agent behavior;
        (3) A JSON file describing the if-then decision rules that govern agent behavior.


        Your goals are to:
        - Produce a complete and structured model specification that can be directly implemented in code, ensuring that all necessary components for implementation are included;
        - Ensure that no new information or assumptions are introduced beyond what appears in the provided files.

        Follow these steps carefully:

        1. Identify all agent types, their attributes, and actions implied by the rules.
        2. Define the environment (e.g., grid, network) and its parameters.
        3. Describe the behavioural rule and the decision-making process in detail, including the if-then rules, and how variables are updated.
        4. Summarize model-level mechanisms such as feedback loops or external influences.
        5. List all simulation parameters, including agent population size, time steps, constants, and sensitivity parameters.
        6. Make sure that all variables and parameters have a reasonable default value and range (or domain).
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
            "default_value_variable_name_1": "a reasonable default value",
            "range_variable_name_1": "expected numerical range or domain",
            "param2": "description of parameter 2",
            "default_value_variable_name_2": "a reasonable default value",
            "range_variable_name_2": "expected numerical range or domain",
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
            "if_then_rules": [
            {
                "rule": "IF condition THEN action",
                "explanation": "social or behavioral reasoning behind the rule"
                "variables_involved": ["list of variables that are part of this rule"],
                "update_effect": "how each variable is updated when this rule is triggered (e.g., increase, decrease, set to a specific value)"
            }
            ]
        },
        "simulation_parameters": {
            "num_agents": "number of agents in the model",
            "time_steps": "number of iterations to simulate",
            "constants": {
            "constant_name_1": "description of what this constant does",
            "default_value_constant_name_1": "a reasonable default value of this constant",
            "range_constant_name_1": "expected numerical range or domain of this constant",
            "constant_name_2": "description of what this constant does",
            "default_value_constant_name_2": "a reasonable default value of this constant",
            "range_constant_name_2": "expected numerical range or domain of this constant"
            }
        },
        "description_of_the_model": "A concise, academic summary (4-5 sentences) describing how the model operates and what emergent phenomena it captures."
        }
        """
        user_prompt = f"""Here is the input problem context: {problem_definition},
        here are the variables that drive agent behavior: {json.dumps(variables, indent=2)}, 
        and here are the if-then decision rules that govern agent behavior: {json.dumps(decision_rules, indent=2)}.    
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
        mechanism = self._safe_json_load(pb)
        return mechanism

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
            6. Make sure that all variables and parameters have a reasonable default value and range (or domain).
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
                "default_value_variable_name_1": "a reasonable default value",
                "range_variable_name_1": "expected numerical range or domain",
                "param2": "description of parameter 2",
                "default_value_variable_name_2": "a reasonable default value",
                "range_variable_name_2": "expected numerical range or domain",
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
                    "default_value_variable_name_1": "a reasonable default value",
                    "range_variable_name_1": "expected numerical range or domain",
                    "variable_name_2": "descrpition of how the value of this variable can be obtained and updated",
                    "default_value_variable_name_2": "a reasonable default value",
                    "range_variable_name_2": "expected numerical range or domain"
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
                "default_value_constant_name_1": "a reasonable default value of this constant",
                "range_constant_name_1": "expected numerical range or domain of this constant",
                "constant_name_2": "description of what this constant does",
                "default_value_constant_name_2": "a reasonable default value of this constant",
                "range_constant_name_2": "expected numerical range or domain of this constant"
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
    def save_odd_to_wordfile(self, text, file_path):
        """
        Save a raw ODD string to a Word file while preserving structure.
        """
        doc = Document()
        doc.add_heading("ODD Model Description", level=1)

        # Split into lines
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                doc.add_paragraph("")  # preserve empty line
                continue
            doc.add_paragraph(line)
        doc.save(file_path)

    def ODD_formatter(self, problem_context:str, preliminary_model:str, secondary_model:str = None):
        """
        This function will format the model description into ODD format, which is a standard format for describing agent-based models.
        """
        with open(problem_context, "r", encoding="utf-8") as file:
            problem_definition = file.read()
        
        system_prompt = """
        You are a computational social scientist specializing in agent-based modeling (ABM).
        Your task is to convert a detailed mechanistic model description into the ODD (Overview, Design concepts, Details) format, which is a standard for describing agent-based models.
        
        You will be provided with:
        (1) A problem context file describing the research question, agent attributes, and environmental setup;
        (2) A JSON file describing the agent decision context, behavioral rules, variables, thresholds, and model parameters.
        
        Your goals are to:
        - Produce a complete and structured model specification in ODD format, ensuring that all necessary components for implementation are included;
        - Ensure that no new information or assumptions are introduced beyond what appears in the provided files.

        Your output should strictly follow the ODD format, which includes the following sections:
        1. Purpose: a concise statement of the model's overall objective and research question.
        2. Entities, state variables, and scales: a detailed description of the agents, their attributes, the environment, and the temporal and spatial scales of the model.
        3. Process overview and scheduling: a step-by-step outline of the processes that occur in each time step, including the order of agent actions and interactions. When and how are state variables updated?
        4. Design concepts: an explanation of the key design principles and mechanisms that drive the model, such as emergence, adaptation, learning, and interaction patterns.
            a.Emergence: Which key model results or outputs are modeled as emerging from the adaptive decisions and behaviors of agents.
            b.Adaptation: How do agents adapt their behavior based on their experiences or changes in the environment?
            c.Learning: Do agents learn from their interactions or outcomes? If so, how is this learning process modeled?
            d.Objectives: What are the goals or objectives that guide agent behavior? Are they maximizing utility, following heuristics, or something else?
            e.Prediction: if an agent's adaptive traits or learning procedures are based on estimating future consequences of decisions, how do agents predict the future conditions (either environmental or internal) they will experience?
            f.Sensing: What information do agents have access to when making decisions? Do they have perfect information about their environment, or do they rely on local perceptions or heuristics?
            g.Interaction: How do agents interact with each other and with the environment? Are interactions local or global, and what is the nature of these interactions (e.g., competition, cooperation, communication)?
            h.Stochasticity: What role does randomness play in the model? Are there stochastic elements in agent decision-making, interactions, or environmental changes?
            i.Collectives: Do the individuals form or belong to aggregations that affect, and are affected by, the individuals?
            j.Observation: What data or outputs are collected from the model, and how do they relate to the research question?
        5. Initialization: a description of how the model is initialized (t = 0), including the initial conditions of agents and the environment.
        6. Input data: a description of any external data that is used to drive the model (if there is), including how it is incorporated and its role in the model dynamics.
        7. Submodels: a detailed description of the submodels that govern specific processes or behaviors in the model, including any mathematical equations or decision rules (if there is any).

        An example of the expected output format can be:
        "Specifically, we are addressing the following questions: [purpose]. The model includes the following entities [entities]. 
        They are characterized by the following state variables [state variables]. 
        The spatial and temporal resolution and extent are [temporal and spatial scales]. 
        The most important design concepts of the model are [all relevant items inthe design concepts].
        The model is initialized with [initialization]. 
        Model dynamics are driven by input data representing [input data description]. (only if there is input data)
        We also include the following submodels to capture key processes in the system: [submodel overview]. (only if there is submodel)
        "
        Export the model description in academic text format that follows the ODD template, ensuring that each section is clearly labeled and contains the relevant information extracted from the provided files.
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
        return pb

    def new_pipeline(self,file_path: str, save_path:str):
            print("Step 1: Brainstorming model ideas based on problem definition...")
            mechanistic_model = self.generate_decision_rule(file_path)
            print(json.dumps(mechanistic_model, indent=2))

            new_model = None
            flag = input("Do you want to propose any new variable based on the current design, or modify the variables proposed by the LLM? (y/n)")
            while flag == "y":
                user_input = input("Please provide name and a short definition of each new variable")
                print("Step2: Generating new model ideas based on your input")
                new_model = self.human_in_the_loop(json.dumps(mechanistic_model, indent=2), user_input)
                print(json.dumps(new_model, indent=2))
                flag = input("Do you want to propose any new variable based on the current design, or modify the variables proposed by the LLM? (y/n)")
            
            print("Step 3: Summarizing the modelling ideas...")
            if new_model:
                summary = self.export_descriptive_model(file_path, mechanistic_model, new_model) # code version
                odd = self.ODD_formatter(file_path, mechanistic_model, new_model) # ODD version
                print(odd)
            else:
                summary = self.export_descriptive_model(file_path, mechanistic_model) # code version
                odd = self.ODD_formatter(file_path, mechanistic_model) # ODD version
                print(odd)
            
            print("Step 4: Export the model...")
            if summary:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)  # save json version for code implementation
                with open(save_path.replace(".json", ".docx"), "w", encoding="utf-8") as f:
                    self.save_odd_to_wordfile(odd, save_path.replace(".json", ".docx"))  # save ODD version for documentation
    
    def test_pipeline(self, file_path: str, save_path:str):
        """Demonstrate how thw flow works linearly"""
        print("Step 1: Generating variables")
        variables = self.decision_rule_variables(file_path)
        print(json.dumps(variables, indent=2))

        print("Step 2: Generating decision rules")
        decision_rules = self.decision_rule_designer(file_path, variables)
        print(json.dumps(decision_rules, indent=2))

        print("Step 3: Summarizing the mechanistic model")
        mechanistic_model = self.mechanism_translation(file_path, variables, decision_rules)
        print(json.dumps(mechanistic_model, indent=2))

        print("Step 4: Export the model")
        odd = self.ODD_formatter(file_path, mechanistic_model) # ODD version
        print(odd)

        if mechanistic_model:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(mechanistic_model, f, indent=2, ensure_ascii=False)  # save json version for code implementation
            with open(save_path.replace(".json", ".docx"), "w", encoding="utf-8") as f:
                self.save_odd_to_wordfile(odd, save_path.replace(".json", ".docx"))  # save ODD version for documentation
