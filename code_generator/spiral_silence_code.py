import random
import numpy as np
import matplotlib.pyplot as plt
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class HumanAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_speaking = False
        self.perceived_public_opinion = random.uniform(0.0, 1.0)
        self.social_connection_strength = random.uniform(0.0, 1.0)
        self.fear_of_isolation = random.uniform(0.0, 1.0)
        self.opinion_stability = random.uniform(0.0, 1.0)

    def decide_to_speak(self):
        beta = self.model.beta
        gamma = self.model.gamma
        influence_threshold = self.model.influence_threshold
        isolation_threshold = self.model.isolation_threshold

        decision_influence = (beta * self.perceived_public_opinion +
                              (1 - beta) * self.social_connection_strength +
                              gamma * self.opinion_stability)

        if decision_influence > influence_threshold:
            self.is_speaking = True
        elif self.fear_of_isolation > isolation_threshold:
            self.is_speaking = False
        else:
            self.is_speaking = False


class LLMAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_speaking = True
        self.opinion = random.choice([0, 1])  # Fixed opinion value of either 0 or 1

    def speak(self):
        return self.opinion


class SpiralOfSilenceModel(Model):
    def __init__(self, num_human_agents, num_llm_agents, width, height):
        super().__init__()  # Explicitly initialize the Model class
        self.num_agents = num_human_agents + num_llm_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.beta = 0.5
        self.gamma = 0.5
        self.influence_threshold = 0.5
        self.isolation_threshold = 0.7
        self.collected_messages = []

        # Create Human Agents
        for i in range(num_human_agents):
            agent = HumanAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Create LLM Agents
        for i in range(num_human_agents, num_human_agents + num_llm_agents):
            agent = LLMAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"is_speaking": "is_speaking"}
        )

    def step(self):
        self.collected_messages = []
        for agent in self.schedule.agents:
            if isinstance(agent, HumanAgent):
                agent.decide_to_speak()
            elif isinstance(agent, LLMAgent):
                self.collected_messages.append(agent.speak())

        # Media System Processing
        if self.collected_messages:
            average_opinion = sum(self.collected_messages) / len(self.collected_messages)
            public_opinion_summary = average_opinion
            for agent in self.schedule.agents:
                agent.perceived_public_opinion = public_opinion_summary

        self.datacollector.collect(self)

        self.schedule.step()


class Visualization:
    @staticmethod
    def plot_data(model):
        data = model.datacollector.get_agent_vars_dataframe()
        plt.figure(figsize=(10, 6))
        plt.title("Agents Speaking Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Speaking Agents")
        speaking_counts = data['is_speaking'].astype(int).groupby(data.index).sum()  # Sum over time steps
        plt.plot(speaking_counts.index, speaking_counts.values, label='Speaking Agents')  # Fixed sum calculation
        plt.legend()
        plt.savefig("output_plots/speaking_agents_over_time.png")
        plt.show()


def main():
    if not os.path.exists("output_plots"):
        os.makedirs("output_plots")

    model = SpiralOfSilenceModel(num_human_agents=50, num_llm_agents=50, width=10, height=10)
    for i in range(10):
        model.step()

    Visualization.plot_data(model)


if __name__ == "__main__":
    main()