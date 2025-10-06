# test the model construct assistant here
from construction_agent import ModelConstructor

# example with shelling segregation model
test_model = ModelConstructor("gpt-4o-mini")
generated_rules = test_model.model_construction_pipeline(
    file_path="Example prompting template.txt"
)
print(generated_rules)
