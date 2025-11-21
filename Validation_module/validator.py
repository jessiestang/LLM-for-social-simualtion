import re
from openai import OpenAI
import json
import os, base64, hashlib

class LLMContext:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # set your API key in environment variable
        self.model_name = model_name
        self.history = [
            {
                "role": "system", "content": "You are a helpful assistant that helps validate computational social science simulation models based on their outputs and conceptual models."
                }
        ]

        self.conceptual_model_str = None
        self.conceptual_model_hash = None

    def set_conceptual_model(self, conceptual_model: dict):
        """Store a canonical serialized conceptual model and its hash.
        Accepts either a Python dict or a filepath (string) pointing to a JSON file.
        Avoids re-sending the full model to the LLM if unchanged across agent calls.
        """
        # if a path to a file is provided, try to load it
        if isinstance(conceptual_model, str) and os.path.exists(conceptual_model):
            with open(conceptual_model, "r", encoding="utf-8") as fh:
                conceptual_model = json.load(fh)

        model_str = json.dumps(conceptual_model, sort_keys=True)
        model_hash = hashlib.sha256(model_str.encode("utf-8")).hexdigest()

        if model_hash != self.conceptual_model_hash: 
            # ensure there is only one conceptual-model system message in history
            self.conceptual_model_str = json.dumps(conceptual_model, indent=2)
            self.conceptual_model_hash = model_hash
            self.history = [m for m in self.history if m.get("meta") != "conceptual_model"]
            self.history.append({
                "role": "system",
                "content": f"Conceptual model (JSON):\n{self.conceptual_model_str}",
                "meta": "conceptual_model",
            })

    def build_messages(self, messages: list):
        """Return a messages list combining the stored history and the provided messages."""
        # copy history to avoid accidental mutation
        combined = list(self.history) + list(messages)
        # sanitize messages: the API expects only keys like 'role', 'content', and optionally 'name'
        sanitized = []
        for m in combined:
            if not isinstance(m, dict):
                continue
            sanitized_msg = {k: v for k, v in m.items() if k in ("role", "content", "name")}
            if "role" in sanitized_msg and "content" in sanitized_msg:
                sanitized.append(sanitized_msg)
        return sanitized

    def chat(self, messages: list, temperature: float = 0.2, max_tokens: int = 1000):
        """Call the underlying OpenAI client using the combined messages."""
        built = self.build_messages(messages)
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=built,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        


class ModelValidation():
    def __init__(self, model_name="gpt-4o-mini"): # the model needs to be able to accept image as input
        # Use a shared LLMContext so multiple agent methods share the same conceptual-model memory
        self.llm_context = LLMContext(model_name=model_name)
        self.model_name = model_name

    def output_analysis(self, image_path, conceptual_model):
        """
        This function sends the image output from the simulation, together with the conceptual model to LLM.
        The LLM agent will help analyzing whether the simulation output aligns with the conceptual model.
        It will give suggestions on the model based on the analysis.
        """
        # read a batch of images from a directory
        image_contents = []
        for file in os.listdir(image_path):
            if file.endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(image_path, file), "rb") as img_file:
                    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })
        
        
        # ensure the shared context contains the conceptual model (no-op if unchanged)
        self.llm_context.set_conceptual_model(conceptual_model)

        # prompting the LLM (keeps model-specific system instructions local to this call)
        system_prompt = """
        You are an expert in computational social science simulation.
        You will be presented with some image outputs from a simulation; refer to the stored conceptual model.
        Your task is to analyze whether the simulation output aligns with the conceptual model.
        If there are any discrepancies, provide suggestions on how to improve the model.
        You may be given images in these following types:
        - Heatmaps or grids: You can analyze the distribution patterns, density, and clustering of entities.
        - Time-series plots: You can analyze trends, fluctuations, and periodicity over time.
        - Network graphs: You can analyze the connectivity, centrality, and community structures.
        - Histograms or bar charts: You can analyze the frequency distributions and comparative metrics.
        Always refer to the conceptual model when analyzing the outputs.
        Output your analysis in this format (JSON only):
        {
        "Plot description": "describe the plot type and key patterns",
        "Analysis": "detailed analysis of how the output aligns or deviates from the conceptual model",
        "Suggestions": "specific suggestions for improving the model"
        }
        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.
        Start your response directly with { and end with }.
        """

        text_part = {
            "type": "text",
            "text": (
                "Here are the simulation output images. Refer to the stored conceptual model for guidance."
            )
        }

        user_prompt = [text_part] + image_contents

        # call the shared LLM context
        LLM_response = self.llm_context.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        # parse the response
        response = LLM_response.choices[0].message.content
        try:
            feedback = json.loads(response)
            return json.dumps(feedback, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return response

        
    def evaluation_suggestion(self, analysis, conceptual_model):
        """
        This function will give specific suggestions on how to evaluate the model,
        based on the conceptual model and the output analysis from LLM.
        """
        conceptual_model = json.dumps(conceptual_model, indent=2)
        output_analysis = json.dumps(analysis, indent=2)

        # prompt the LLM
        system_prompt = """
        You are an expert in computational social science simulation.
        Now you will be presented with a conceptual model description and an analysis of simulation outputs.
        Your task is to provide suggestions on how we can evaluate the model. from the perspective of stochasticity control, parameter sensitivity analysis, and statistical tests.
        Your suggestions should include the following sections:
        1. Stochasticity Control: How to control for randomness in the simulation runs.
        Be specific about the methods to use, the number of simulation runs needed, and the reasoning behind your suggestions.
        Also state what you expect to observe if the stochasticity is well controlled.
        Write in detailed steps on how to implement it.
        An example can be: "To control for stochasticity, we can use a Monte Carlo approach by running the simulation [a number] times with different random seeds.
        This will help us capture the variability in the outputs due to randomness. We can then compute the mean and standard deviation of key output metrics across these runs to assess stability.
        If stochasticity is well controlled, we expect the standard deviation of output metric X to be below a certain threshold, say [a number].
        We can verify this by plotting the distribution of output metric X across all runs and checking for convergence."

        2. Parameter Sensitivity Analysis: Which parameters to vary and how to assess their impact on the outputs.
        Suggested parameters should be from the conceptual model. DO NOT invent any new parameter yourself here.
        Be very specific about the range of values for analysis of each parameter, based on your internal reasoning.
        Be very specific about the approach to the sensitivity analysis. Do not just mention the name, but provide reason on why this approach,
        and detailed steps on how to implement it.
        Also give some suggestions on how to evaluate the impact of those parameters on the outputs.
        An example can be: "To analyze the sensitivity of parameter X, which ranges from [a number] to [a number], we can use a one-at-a-time (OAT) approach. 
        We will vary parameter X in increments of [a number] while keeping other parameters constant, and run [a number] simulation iterations for each value to observe the changes in output metric Y.
        We will then plot the results to visualize the sensitivity."

        3. Uncertainty Quantification: What metrics and statistical methods can be used to quantify uncertainty in the outputs, and how to compute them.
        Give reasons about why those metrics are suitable for this model, and how to compute them based on the conceptual model structure.
        An example can be: "To quantify uncertainty in output metric Z, we can compute the [a percentage] confidence interval using bootstrapping. 
        This involves resampling the simulation outputs with replacement [a number] times and calculating the interval from the resulting distribution. 
        This method is suitable because it does not assume a specific distribution for the outputs, which aligns with the stochastic nature of the model."

        4. Experimental Design: Based on the above three sections, provide a concise experimental design plan summarizing the key steps to implement the evaluation.
        Give concrete steps on how to carry out the experiment, including the number of simulation runs, parameter settings, and analysis methods.
        An example can be:" To implement the evaluation, we will first conduct stochasticity control by running the simulation [a number] times with varied random seeds.
        Next, we will perform parameter sensitivity analysis on parameters X and Y using the OAT approach, varying each in specified ranges while keeping others constant.
        Finally, we will quantify uncertainty in output metric Z using bootstrapping to compute the [a percentage] confidence interval.
        The entire experiment will involve a total of [a number] simulation runs."
        Your output should be in this format:
        {
        "Stochasticity Control": "detailed suggestions",
        "Parameter Sensitivity Analysis": "detailed suggestions",
        "Uncertainty Quantification": "detailed suggestions",
        }

        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.
        Start your response directly with { and end with }.
        """

        user_prompt = f"""
        Here is the analysis of the simulation outputs: {output_analysis}.
        Based on the conceptual model and output_analysis,
        please provide your evaluation suggestions following the instructions in the system prompt.
        """
        self.llm_context.set_conceptual_model(json.loads(conceptual_model))

        LLM_response = self.llm_context.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        # parse the response
        # parse the response to extract JSON
        response = LLM_response.choices[0].message.content
        try:
            suggestions = json.loads(response)
            return json.dumps(suggestions, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return response

    def rq_driven_experimental_design(self, conceptual_model): #TODO: maybe need to move this part to another module
        """ 
        This function will help design the experiment plan keep the ABM fixed and explore theoretical dynamics.
        """
        conceptual_model = json.dumps(conceptual_model, indent=2)

        # prompt the LLM
        system_prompt = """
        You are an expert in experimental design for computational social science simulations.
        Your task is to design an experiment plan to explore theoretical dynamics of a given conceptual model.
        The conceptual model will be provided as a json object.
        Your output should include the following sections:
        1. Key variables and dynamics: Identify the key variables and dynamics in the conceptual model that are relevant to the research questions.
        2. Hypotheses: Formulate clear hypotheses about the expected behaviors and outcomes based on the conceptual model.
        3. Experimental conditions: For each hypothesis, define the experimental conditions, including parameter settings, initial conditions, and any interventions to be tested.
        4. Data collection and analysis: For each experiment, outline the data collection methods and analysis techniques to be used to evaluate the hypotheses.
        Your output should be in this format:
        {
        "Key Variables and Dynamics": "detailed description in natural language",
        "Hypotheses": [Hypothesis1, Hypothesis2],
        "Experimental Conditions": "detailed conditions in natural language",
        "Data Collection and Analysis": "detailed methods in natural language"
        }

        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.
        Start your response directly with { and end with }.
        """

        user_prompt = f"""
        Please provide your experimental design following the instructions in the system prompt.
        """

        # call the LLM
        # use the shared LLM context and ensure conceptual model is set
        self.llm_context.set_conceptual_model(json.loads(conceptual_model))
        experiment = self.llm_context.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        # parse the response
        response = experiment.choices[0].message.content
        try:
            feedback = json.loads(response)
            return json.dumps(feedback, indent=2)
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return response

    
    def dataset_finding(self):
        """
        This function will search for relevant datasets for model validation using DuckDuckGo API or other dataset repositories.
        The search function will be implemented as an external tool later.
        """

        pass
    
    def run_pipeline(self, image_path, conceptual_model):
        """
        Run this pipeline to perform output analysis and model validation
        """
        print("You must provide the detailed directory path where you store the simulation output images, and the conceptual model as a json object.")
        # if type(conceptual_model) != dict:
           #  raise ValueError("The conceptual model must be provided as a json object (Python dict).")
        if not os.path.exists(image_path):
            raise ValueError("The provided image path does not exist.")
        
        print("Starting output analysis...")
        analysis = self.output_analysis(image_path, conceptual_model)
        print(f"Output analysis completed: {analysis}")
        print("Starting validation suggestion generation...")
        suggestions = self.evaluation_suggestion(analysis, conceptual_model)
        print(f"Validation suggestions generated: {suggestions}")
        experiment = self.rq_driven_experimental_design(conceptual_model)
        print(f"Experimental design generated: {experiment}")

        #TODO: still need to add human in the loop for review
        