# run the validator with this file
from validator import ModelValidation
import os
import json

image_path = r"E:\LLM_for_abm\LLM-for-social-simualtion\code_generator\output_plots"
print(image_path)
conceptual_model = os.path.join ("..", "model_construct_assistant", "Conceptual Model.json")    
vali = ModelValidation(model_name="gpt-4o-mini")
vali.run_pipeline(image_path, conceptual_model)