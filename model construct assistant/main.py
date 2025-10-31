# test the model construct assistant here
from construction_agent import ModelConstructor

# example with shelling segregation model
test_model = ModelConstructor("gpt-4o-mini")
brainstormed_rules = test_model.model_brainstorming(
    problem_context="Example prompting template2.txt"
)
print(brainstormed_rules)


"""generated_rules = test_model.model_construction_pipeline(
    file_path="Example prompting template2.txt", save_path="Conceptual Model.json"
)
print(generated_rules)"""
