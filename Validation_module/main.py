# run the validator with this file
from validator import ModelValidation
import os
import json

#image_path = r"E:\LLM_for_abm\LLM-for-social-simualtion\code_generator\output_plots"
#print(image_path)

conceptual_model = os.path.join ("..", "model_construct_assistant", "Shelling_model.json")
model_interface = "model_interface.json"
model_code = os.path.join("..", "code_generator", "generated_shelling_model.py")
output_path = os.path.join("..", "Validation_module", "evaluation_strategy.py")
vali = ModelValidation(model_name="gpt-4o-mini")
suggestion = vali.run_pipeline2(model_code=model_code, conceptual_model=conceptual_model, model_interface=model_interface, output_path=output_path)
