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
   
    def _ensure_dict(self, val):
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            # Strip markdown fences if present
            cleaned = re.sub(r"```json|```", "", val).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Handle double-escaped strings
                try:
                    return json.loads(cleaned.encode().decode('unicode_escape'))
                except Exception:
                    raise ValueError(f"Cannot parse as JSON: {cleaned[:200]}")
        raise ValueError(f"Expected dict or JSON string, got {type(val).__name__}: {str(val)[:200]}")
    
    def evaluation_suggestion(self, conceptual_model:object):
        """
        This function will give specific suggestions on how to evaluate the model,
        based on the conceptual model and the output analysis from LLM.
        """
        conceptual_model = self._ensure_dict(conceptual_model)
        conceptual_model = json.dumps(conceptual_model, indent=2)
        # output_analysis = json.dumps(analysis, indent=2)

        # prompt the LLM
        system_prompt = """
        You are an expert in computational social science simulation.
        Now you will be presented with a conceptual model description.
        Your task is to provide suggestions for evaluating the model.
        Your suggestions should include the following sections:
        1. Stochasticity Control: How to control for randomness in the simulation runs.
        Be specific about the methods to use, the number of simulation runs needed, and the reasoning behind your suggestions.
        Also state what you expect to observe if the stochasticity is well controlled.
        Write in detailed steps on how to implement it.
        An example can be: "To control for stochasticity, we can use a [approach name] approach by running the simulation [a number] times with different random seeds.
        This will help us capture the variability in the outputs due to randomness. We can then compute the mean and standard deviation of key output metrics across these runs to assess stability.
        If stochasticity is well controlled, we expect the standard deviation of output metric X to be below a certain threshold, say [a number].
        The metric X is defined as [definition based on conceptual model or internal reasoning], and can be measured by [detailed steps].
        We can verify this by plotting [suggestions for visualization]. This approach is suitable because [a reason]."

        2. Parameter Sensitivity Analysis: Which parameters to vary and how to assess their impact on the outputs.
        Suggested parameters should be from the conceptual model. DO NOT invent any new parameter yourself here.
        Be very specific about the range of values for analysis of each parameter, based on your internal reasoning.
        Be very specific about the approach to the sensitivity analysis. Do not just mention the name, but provide reason on why this approach,
        and detailed steps on how to implement it. If the output metrics is not mentioned in the conceptual model, define the conceptual model and explain how that would be measured.
        Also give some suggestions on how to evaluate the impact of those parameters on the outputs.
        An example can be: "To analyze the sensitivity of parameter X, which ranges from [a number] to [a number], we can use a [approach name] approach. 
        We will vary parameter X in increments of [a number] while keeping other parameters constant, and run [a number] simulation iterations for each value to observe the changes in output metric Y.
        The output metric Y is defined as [definition based on conceptual model or internal reasoning], and can be measured by [detailed steps].
        We will then visualize the sensitivity by plotting [suggestions for visualization]. This approach is suitable because [a reason]."

        3. Uncertainty Quantification: What metrics and statistical methods can be used to quantify uncertainty in the outputs, and how to compute them.
        Give reasons about why those metrics are suitable for this model, and how to compute them based on the conceptual model structure. 
        If the output metrics is not mentioned in the conceptual model, define the conceptual model and explain how that would be measured.
        An example can be: "To quantify uncertainty in output metric Z, we can compute the [a percentage] confidence interval using [a statistical method]. 
        This involves resampling the simulation outputs with replacement [a number] times and calculating the interval from the resulting distribution. 
        This method is suitable because [a reason]. Here the output metric Z is defined as [definition based on conceptual model or internal reasoning].
        This metric can be computed by [detailed steps]."

        4. Cross-condition Comparison: Define 2-3 theoretically meaningful experimental conditions that represent different real-world scenarios, and compare model behavior across them.
        Conditions should differ in ONE parameter at a time to allow causal interpretation. For each condition:
        - State the theoretical motivation for this condition
        - Specify exactly which parameter changes and to what value
        - State the expected direction of effect and why
        - Define how to statistically compare outcomes across conditions
        An example can be: " "We compare three [variable] conditions: low ([variable value]), medium ([variable value]), and high ([variable value]).
        The expected effect is that [your predicion]. We measure [outcome variable] across [number] runs per condition.
        We and compare outputs in different conditions using [statsitical test].""


        Your output should be in this format:
        [
        {"strategy_id": "1",
        "strategy_type": "Stochasticity Control",
        "description": "detailed suggestions"},

        {"strategy_id": "2",
        "strategy_type": "Parameter Sensitivity Analysis",
        "description": "detailed suggestions"},

        {"strategy_id": "3",
        "strategy_type": "Uncertainty Quantification",
        "description": "detailed suggestions"},

        {"strategy_id": "4",
        "strategy_type": "Cross-condition Comparison",
        "description": "detailed suggestions"},
        
        ]

        CRITICAL: Output ONLY valid JSON. Do not include any text before or after the JSON. Do not wrap in markdown code blocks.
        Start your response directly with [ and end with ].
        """

        user_prompt = f"""
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
        response = LLM_response.choices[0].message.content
        rules = self._safe_json_load(response)
        return rules
    
    def evaluation_code_generator(self, evaluation_suggestions:object, model_interface:object, output_path:str):
        """
        Generate evaluation code aligned with an existing simulation model.
        """
        # load conceptual model and model interface
        evaluation_suggestions = self._ensure_dict(evaluation_suggestions)
        model_interface = self._ensure_dict(model_interface)

        evaluation_suggestions = json.dumps(evaluation_suggestions, indent=2)
        model_interface = json.dumps(model_interface, indent=2)

        system_prompt = """
        You are an expert in computational social science and Python programming.
        Generate evaluation code that aligns with an existing agent-based model.
        You will be provided with a model interface description, and evaluation suggestions.
        You have two tasks:
        1. Generate Python code that implements the evaluation suggestions using the provided model interface and the model code.
        Do NOT modify model internals.
        Do NOT modify existed classes or functions in the model code.
        Only vary parameters, execute runs, and analyze outputs.
        Do NOT provide any explanations or notes outside the code. Just provide the code.

        2. Re-inspect the generated code to ensure it adheres to the model interface and evaluation suggestions.
        Make sure that all required functions and classes from the model interface are properly utilized in the generated code.
        Make sure that all relevant dependencies are imported.
        Your final output should be ONLY the complete Python code file.
        If you need to make any corrections, do so directly in the code.
        Your output should be a complete Python code file that can be run independently.
        """

        user_prompt = f"""Model interface:{model_interface}, Evaluation task:{evaluation_suggestions}
        Generate Python code that implements this evaluation.
        """

        response = self.llm_context.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        code = response.choices[0].message.content
        code = code.replace("```python\n", "").strip() # remove unnecessary markdown formatting if any
        code = code.replace("```", "").strip()
        with open(output_path, "w") as f: # export to a python file
            f.write(code)

        return code
