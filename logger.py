import json
import os
from datetime import datetime
from typing import Any, Dict

class Logger:
    def __init__(self, run_name: str, base_dir: str = "runs"):
        """
        Initialize a record for one model run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # get the current time
        self.run_id = f"{run_name}_{timestamp}" # unique run id each model run
        self.run_dir = os.path.join(base_dir, self.run_id)
        self._create_structure()
        self._write_metadata(
            {   "run_id": self.run_id,
                "run_name": run_name,
                "timestamp": timestamp
            }
        )

        def _create_structure(self):
            """
            Create the directory structure for logging
            """
            subdirs = ["prompts", "outputs", "human_decisions", "executions_metadata"]
            for sub in subdirs:
                os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True) # create subdirectories for each type of log
        
        def _write_metadata(self, metadata):
            """
            This function writes the initial metadata for the run into a json file
            """
            meta_data_path = os.path.join(self.run_dir, "metadata.json") # initializing writing path
            with open(meta_data_path, "w") as f:  # write metadata
                json.dump(metadata, f, indent=4)
        
        def update_metadata(self, updates):
            """
            This function updates the metadata json file with new information
            """
            meta_data_path = os.path.join(self.run_dir, "metadata.json") # retrieve the path
            with open(meta_data_path, "r") as f:
                metadata = json.load(f) # read the metadata
            metadata.update(updates) # update the metadata
            with open(meta_data_path, "w") as f:
                json.dump(metadata, f, indent = 4)

        def log_prompt(self, system_prompt, user_prompt, parameters):
            """
            This function logs the system prompt, user prompt, and other parameters when calling LLM
            """
            records = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "parameters": parameters
            }  # example parameters include: model name, temperature and max tokens
            prompt_log_path = os.path.join(self.run_dir, "prompt.json")
            with open(prompt_log_path, "w") as f:
                json.dump(records, f, indent=4)

        def log_output(self, llm_output):
            """
            This function logs the output from the LLM
            """
            output_log_path = os.path.join(self.run_dir, "outputs.json")
            with open(output_log_path, "w") as f:
                json.dump(llm_output, f, indent = 4)

        def log_human_decision(self, decision_type, content):
            """
            This function logs the human interaction with the LLM
            """
            record = {
                "decision_type": decision_type,
                "content": content
            } # example decision types may include: varaible addition, variable modification, evaluation selection
            human_log_path = os.path.join(self.run_dir, "human_decision.json")
            with open(human_log_path, "w") as f:
                json.dump(record, f, indent = 4)

        

