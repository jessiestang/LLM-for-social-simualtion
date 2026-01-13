# test the model construct assistant here
from construction_agent import ModelConstructor

# example with shelling segregation model
test_model = ModelConstructor("gpt-4o-mini")
"""generated_rules = test_model.model_construction_pipeline(
    file_path="shelling_segregation.txt", save_path="shelling_model.json"
)"""
generated_rules = test_model.new_pipeline(file_path="shelling_segregation.txt", save_path = "Shelling_model.json")
print(generated_rules)
