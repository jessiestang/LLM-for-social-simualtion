from openai import OpenAI
import json
import os, base64

class ModelValidation():
    def __init__(self, model_name="gpt-4o-mini"): # the model needs to be able to accept image as input
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # set your API key in environment variable
        self.model_name = model_name 

    def output_analysis(self, image_path, conceptual_model):
        """
        This function sends the image output from the simulation, together with the conceptual model to LLM.
        The LLM agent will help analyzing whether the simulation output aligns with the conceptual model.
        It will give suggestions on the model based on the analysis.
        """
        # read a batch of images from a directory
        image = []
        for file in os.listdir(image_path):
            if file.endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(image_path, file), "rb") as img_file:
                    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    image.append({"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"})
       
        
        # prompting the LLM
        system_prompt = """
        You are an expert in computational social science simulation.
        You will be presented with some image outputs from a simulation, together with a conceptual model description.
        Your task is to analyze whether the simulation output aligns with the conceptual model.
        If there are any discrepancies, provide suggestions on how to improve the model.
        You may be given images in these following types:
        - Heatmaps or grids: You can analyze the distribution patterns, density, and clustering of entities.
        - Time-series plots: You can analyze trends, fluctuations, and periodicity over time.
        - Network graphs: You can analyze the connectivity, centrality, and community structures.
        - Histograms or bar charts: You can analyze the frequency distributions and comparative metrics.
        Always refer to the conceptual model when analyzing the outputs.
        The conceptual model will be a json file describing the structure and expected behaviors of the simulation.
        Output your analysis in this format:
        {
        "Plot description": [describe the plot type and key patterns]
        "Analysis": [detailed analysis of how the output aligns or deviates from the conceptual model]
        "Suggestions": [specific suggestions for improving the model]
        }
        """

        user_prompt = f"""Here is the conceptual model description:{json.dumps(conceptual_model, indent=2)},
        and here are the simulation output images: {image}.
        Please provide your analysis following the instructons in the system prompt.
        """ 

        # call the LLM
        LLM_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens = 1500,
        )

        # parse the response
        response_content = LLM_response.choices[0].message.content
        try:
            analysis = json.loads(response_content)
        except json.JSONDecodeError:
            analysis = {"error": "Failed to parse LLM response", "response": response_content}

        return analysis
        
    def validation_suggestion(self, analysis, conceptual_model):
        """
        This function will give specific suggestions on how to validate the simulation,
        based on the conceptual model and the output analysis from LLM.
        """
        conceptual_model = json.dumps(conceptual_model, indent=2)
        output_analysis = json.dumps(analysis, indent=2)

        # prompt the LLM
        system_prompt = """
        You are an expert in computational social science simulation.
        Now you will be presented with a conceptual model description and an analysis of simulation outputs.
        Your task is to provide suggestions on how we can validate the simulation. from the perspective of stochasticity control, parameter sensitivity analysis, and statistical tests.
        Your suggestions should include the following sections:
        1. Stochasticity Control: How to control for randomness in the simulation runs.
        2. Parameter Sensitivity Analysis: Which parameters to vary and how to assess their impact on the outputs.
        3. Uncertainty Quantification: What metrics can be used to quantify uncertainty in the outputs, and how to compute them.
        4. Experimental Design: How to design simulation experiments to effectively validate the model.
        5. External Validation: How to demonstrate the validity of the simulation model in real world scenarios, like using a real-world dataset.
        Provide your suggestions in a structured format with clear headings for each section.
        Your output should be in this format:
        {
        "Stochasticity Control": [detailed suggestions],
        "Parameter Sensitivity Analysis": [detailed suggestions],
        "Uncertainty Quantification": [detailed suggestions],
        "Experimental Design": [detailed suggestions],
        "External Validation": [detailed suggestions]}
        """

        user_prompt = f"""
        Here is the conceptual model description: {conceptual_model},
        and here is the analysis of the simulation outputs: {output_analysis}.
        Please provide your validation suggestions following the instructions in the system prompt.
        """

        # call the LLM
        LLM_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens = 1500,
        )

        # parse the response
        response_content = LLM_response.choices[0].message.content
        try:
            suggestions = json.loads(response_content)
        except json.JSONDecodeError:
            suggestions = {"error": "Failed to parse LLM response", "response": response_content}
        
        return suggestions
    
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
        suggestions = self.validation_suggestion(analysis, conceptual_model)
        print(f"Validation suggestions generated: {suggestions}")

        #TODO: still need to add human in the loop for review
        