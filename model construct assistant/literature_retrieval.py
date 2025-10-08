from openai import OpenAI
import os
import json
import requests


class Literature_retrieval:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.openALEX_api = api_key = os.getenv(
            "OPENALEX_API_KEY"
        )  # need to have an openalex api key and set as environment variable

    def keyword_extraction(self, query: str) -> list:
        """
        This llm agent extracts keywords from the problem definition provided by users
        """
        system_prompt = """
        You are a research assistant who specializes in social simulation models.
        You are given a problem definition provided by users.
        Your task is to extract around 5 keywords that best represent the problem definition for literature retrieval.
        Output strictly in JSON format:
        {
            "keywords": ["keyword1", "keyword2", "..."]
        }
        Ensure your response is valid JSON that can be parsed by a JSON parser.

        Example
        The given research question is:
        "We want to model the voting behaviour of citizens in a democratic society.
        Suppose there are two types of voters with different political preferences (+1 and -1).
        Voters interact in a social network where they can influence each other's opinions.
        Each voter decides whether to keep or change their opinin based on that of their neighbours."

        The extracted keywords should be:
        {
            "keywords": ["voting behavior", "computational model", "political behavior simulation", "social network", "agent-based modeling"]
        }
        """
        user_prompt = f""" Given the following definition:{query}, extract around 5 keywords for literature research."""

        # call the LLM model
        llm_model = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        # parse the response to extract JSON
        try:
            response = json.loads(llm_model.choices[0].message.content)
            return response["keywords"]
        except json.JSONDecodeError:
            print("LLM output not valid JSON, here’s raw output:")
            return llm_model.choices[0].message.content

    def literature_research(self, keywords: list):
        """
        This function uses OpenAlex API to extract relevant literature based on keywords.
        """
        papers = 0
        return papers

    def paper_summarization(self, papers):
        """
        This llm agent will read through the retrieved papers and summarize key insights that can inform model construction.
        """
        summaries = 0
        return summaries
