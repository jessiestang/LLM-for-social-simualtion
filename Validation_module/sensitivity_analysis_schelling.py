import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mesa import Model
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
sys.path.append(os.path.abspath( "../code_generator"))
from generated_shelling_model import AgentRacialType1, AgentRacialType2



class SegregationModel(Model):
    def __init__(self, num_agents, width, height, homogeneity_preference):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.homogeneity_preference = homogeneity_preference
        self.unhappy_agents_count = []

        for i in range(self.num_agents // 2):
            agent = AgentRacialType1(i, self)
            agent.homogeneity_preference = self.homogeneity_preference
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        for i in range(self.num_agents // 2, self.num_agents):
            agent = AgentRacialType2(i, self)
            agent.homogeneity_preference = self.homogeneity_preference
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        unhappy_agents = 0
        for agent in self.schedule.agents:
            agent.decide_movement()
            if agent.decision_signal < agent.homogeneity_preference:
                unhappy_agents += 1
        self.unhappy_agents_count.append(unhappy_agents)
        self.schedule.step()

def racial_segregation_index(model):
    same_race_neighbors_count = 0
    total_agents = model.num_agents

    for agent in model.schedule.agents:
        neighbors = model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
        same_race_count = sum(1 for neighbor in neighbors if isinstance(neighbor, type(agent)))
        if same_race_count / len(neighbors) > 0.7:
            same_race_neighbors_count += 1

    return same_race_neighbors_count / total_agents if total_agents > 0 else 0

def evaluate_homogeneity_preference():
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')

    homogeneity_preferences = np.arange(0, 1.1, 0.1)
    segregation_indices = []

    for hp in homogeneity_preferences:
        model = SegregationModel(num_agents=300, width=20, height=20, homogeneity_preference=hp)
        
        for _ in range(500):  # Run for 1000 steps
            model.step()
        
        segregation_index = racial_segregation_index(model)
        segregation_indices.append(segregation_index)

    plt.plot(homogeneity_preferences, segregation_indices, marker='o')
    plt.title("Sensitivity Analysis of Homogeneity Preference")
    plt.xlabel("Homogeneity Preference")
    plt.ylabel("Racial Segregation Index")
    plt.xticks(homogeneity_preferences)
    plt.savefig('output_plots/homogeneity_preference_analysis.png')
    plt.show()

if __name__ == "__main__":
    evaluate_homogeneity_preference()