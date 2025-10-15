# main file to run the LLM module
from coding_agent import CodingAgent
import os

json_path = os.path.join ("..", "model construct assistant", "Conceptual Model.json")
code = CodingAgent("gpt-4o-mini")
generated_code = code.code_generation(json_path = json_path)
print(generated_code)