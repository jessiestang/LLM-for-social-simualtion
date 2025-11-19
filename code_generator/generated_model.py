import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

class UserAgent(Agent):
    """An agent representing a user in the social network."""
    
    def __init__(self, unique_id, model, ideology, attention_level, engagement_level):
        super().__init__(unique_id, model)
        self.ideology = ideology
        self.attention_level = attention_level
        self.engagement_level = engagement_level
        self.followers = set()
        self.reposts = 0

    def step(self):
        # Simulate user actions
        self.interact_with_network()

    def interact_with_network(self):
        neighbors = self.model.grid.get_neighbors(self.unique_id)
        for neighbor in neighbors:
            if self.should_engage(neighbor):
                self.engage_with_content(neighbor)
                self.share_content(neighbor)

    def should_engage(self, neighbor):
        return (self.model.ideological_similarity(self.ideology, neighbor.ideology) and 
                self.attention_level > random.random())

    def engage_with_content(self, neighbor):
        if self.ideology == neighbor.ideology:
            self.reposts += 1

    def share_content(self, neighbor):
        if self.reposts > 0:
            neighbor.followers.add(self)

class SocialModel(Model):
    """A model with some number of agents."""
    
    def __init__(self, N, network_density):
        super().__init__()  # Initialize the Model superclass
        self.num_agents = N
        self.grid = NetworkGrid(nx.erdos_renyi_graph(N, network_density))
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={"ideology": "ideology", "followers": "followers", "reposts": "reposts"}
        )

        # Create agents
        ideologies = ['left_leaning', 'right_leaning', 'neutral']
        for i in range(self.num_agents):
            ideology = random.choice(ideologies)
            attention_level = random.uniform(0, 1)
            engagement_level = random.uniform(0, 1)
            agent = UserAgent(i, self, ideology, attention_level, engagement_level)
            self.schedule.add(agent)
            self.grid.place_agent(agent, i)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def ideological_similarity(self, ideology1, ideology2):
        return ideology1 == ideology2

    def calculate_ei_index(self):
        E, I = 0, 0
        for agent in self.schedule.agents:
            for follower in agent.followers:
                if self.ideological_similarity(agent.ideology, follower.ideology):
                    I += 1
                else:
                    E += 1
        return (E - I) / (E + I) if (E + I) > 0 else 0

    def calculate_gini_coefficient(self, attribute):
        values = [len(agent.followers) if attribute == 'followers' else agent.reposts for agent in self.schedule.agents]
        n = len(values)
        if n == 0:
            return 0
        sorted_values = np.sort(values)
        index = np.arange(1, n + 1)
        return 1 - (2 / n) * (np.sum((n + 1 - index) * sorted_values) / np.sum(sorted_values))

    def calculate_correlations(self):
        followers = np.array([len(agent.followers) for agent in self.schedule.agents])
        reposts = np.array([agent.reposts for agent in self.schedule.agents])
        ideologies = np.array([agent.ideology for agent in self.schedule.agents])
        
        # Convert ideologies to numerical values for correlation
        ideology_map = {'left_leaning': -1, 'neutral': 0, 'right_leaning': 1}
        ideology_numeric = np.vectorize(ideology_map.get)(ideologies)

        correlation_followers = np.corrcoef(ideology_numeric, followers)[0, 1]
        correlation_reposts = np.corrcoef(ideology_numeric, reposts)[0, 1]

        return correlation_followers, correlation_reposts

class Visualization:
    """Class for visualizing the social network."""
    
    @staticmethod
    def plot_network(model):
        G = nx.Graph()
        for agent in model.schedule.agents:
            G.add_node(agent.unique_id, ideology=agent.ideology, followers=len(agent.followers))
            for follower in agent.followers:
                G.add_edge(agent.unique_id, follower.unique_id)

        color_map = {'left_leaning': 'blue', 'neutral': 'gray', 'right_leaning': 'red'}
        node_colors = [color_map[G.nodes[node]['ideology']] for node in G.nodes()]
        node_sizes = [G.nodes[node]['followers'] * 10 for node in G.nodes()]  # Corrected line

        plt.figure(figsize=(12, 12))
        nx.draw(G, node_color=node_colors, node_size=node_sizes, with_labels=True, font_weight='bold', alpha=0.7)
        plt.title("Social Network Visualization")
        plt.savefig("output_plots/social_network.png")
        plt.show()

if __name__ == "__main__":
    model = SocialModel(N=1000, network_density=0.05)  # Changed 'num_agents' to 'N'
    for i in range(10):
        model.step()

    ei_index = model.calculate_ei_index()
    gini_followers = model.calculate_gini_coefficient('followers')
    gini_reposts = model.calculate_gini_coefficient('reposts')
    correlation_followers, correlation_reposts = model.calculate_correlations()

    print(f"E-I Index: {ei_index}")
    print(f"Gini Coefficient of Followers: {gini_followers}")
    print(f"Gini Coefficient of Reposts: {gini_reposts}")
    print(f"Correlation between Partisanship and Number of Followers: {correlation_followers}")
    print(f"Correlation between Partisanship and Repost Activity: {correlation_reposts}")

    Visualization.plot_network(model)