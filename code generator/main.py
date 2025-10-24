# main file to run the LLM module
from coding_agent import CodingAgent, MasterAgent, SubCodeAgents
import os

# load the conceptual model json file and user prompt
json_path = os.path.join ("..", "model construct assistant", "Conceptual Model.json")
with open("user_requirements.txt", "r", encoding="utf-8") as file:
            user_requirements = file.read()
            
# run the coding agent pipeline here
code = CodingAgent("gpt-4o-mini")
generated_code = code.run_pipeline(json_path, user_requirements)
# print(generated_code)

# run the coding agent pipeline here
# master_agent = MasterAgent(json_path,"gpt-4o-mini")
# master_agent.run_pipeline()

