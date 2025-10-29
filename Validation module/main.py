# run the validator with this file
from validator import ModelValidation
import os
import json

image_path = os.path.join(os.path.dirname(__file__),"..", "code generator") # test image path
print(image_path)
conceptual_model = os.path.join ("..", "model construct assistant", "Conceptual Model.json")    
vali = ModelValidation(model_name="gpt-4o-mini")
vali.run_pipeline(image_path, conceptual_model)