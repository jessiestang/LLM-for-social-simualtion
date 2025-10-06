# tools to be implemented later for agent to use
import json
import os


class Tool_box_LLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def literature_search(self, query: str) -> str:  # TODO: to be implemented later
        """
        A tool that allows AI agent to search for relevant literature through API call
        """
        return "No literature yet"

    def ODD_template(self) -> str:
        """
            A tool that translates LLM output rules to ODD format
            The output should be a word file or latex file with the following structure:
        1. Overview
            - Purpose and patterns
            - Entities, state variables and scales
            - Process overview and scheduling
        2. Design concepts
            - Basic principles
            - Emergence
            - Adaptation
            - Objectives
            - Learning
            - Prediction
            - Sensing
            - Interaction
            - Stochasticity
            - Collectives
            - Observation
        3. Details
            - Initialization
            - Input data
            - Submodels

        """
        return "No ODD yet"

    def load_prompting_template(self, template_path: str) -> str:
        """
        A tool that loads prompting template from a file
        """
        with open(template_path, "r") as file:
            template = file.read()
        return template
