import random
import numpy as np
import matplotlib.pyplot as plt
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class SocialMedia(Model):
    def __init__(self, num_agents=100, width=10, height=10):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.media_opinions = []
        self.silent_ratios = {'opinion_0': [], 'opinion_1': []}
        self._media_gap = []  # Changed to private attribute
        
        for i in range(self.num_agents):
            if random.random() < 0.8:
                agent = HumanAgent(i, self)
            else:
                agent = LLMAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"opinion": "opinion", "opinion_similarity_ratio": "opinion_similarity_ratio"},
            model_reporters={"silent_ratio_0": "silent_ratio_0", "silent_ratio_1": "silent_ratio_1", "media_gap": "media_gap"}
        )

    def step(self):
        self.schedule.step()
        self.update_media_opinion()
        self.calculate_silent_ratios()
        self.calculate_media_gap()
        self.datacollector.collect(self)

    def update_media_opinion(self):
        self.media_opinions = [agent.opinion for agent in self.schedule.agents if agent.is_speaking]   
        return self.media_opinions

    def calculate_silent_ratios(self):
        silent_count_0 = sum(1 for agent in self.schedule.agents if agent.opinion == 0 and not agent.is_speaking)
        silent_count_1 = sum(1 for agent in self.schedule.agents if agent.opinion == 1 and not agent.is_speaking)
        num_type_0 = sum(1 for agent in self.schedule.agents if agent.opinion == 0)
        num_type_1 = sum(1 for agent in self.schedule.agents if agent.opinion == 1)
        self.silent_ratios['opinion_0'].append(silent_count_0 / num_type_0)
        self.silent_ratios['opinion_1'].append(silent_count_1 / num_type_1)

    def calculate_media_gap(self):
        # count how many agents are speak for 0 and 1 in opinions list
        self.media_message_0 = sum(1 for opinion in self.media_opinions if opinion == 0)
        self.media_message_1 = sum(1 for opinion in self.media_opinions if opinion == 1)
        if len(self.media_opinions) > 1:
            gap = abs(self.media_message_1 - self.media_message_0)
            self._media_gap.append(gap)

    @property
    def silent_ratio_0(self):
        return self.silent_ratios['opinion_0'][-1] if self.silent_ratios['opinion_0'] else 0

    @property
    def silent_ratio_1(self):
        return self.silent_ratios['opinion_1'][-1] if self.silent_ratios['opinion_1'] else 0

    @property
    def media_gap(self):
        return self._media_gap[-1] if self._media_gap else 0  # Accessing the private attribute

    @property
    def media_gap_series(self):
        return self._media_gap

class HumanAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = random.choices([0, 1], weights=[0.6, 0.4])[0]
        self.perceived_public_opinion = 0
        self.social_isolation_fear = random.uniform(0, 1)
        self.confidence_level = random.uniform(0, 1)
        self.perceived_local_opinion = 0
        self.is_speaking = False
        self.opinion_similarity_ratio = 0.0
        self.permanently_silent = False  # Once silent, always silent

    def step(self):
        # get public opinion from media
        self.calculate_public_opinion()
        
        # get local opinion from neighbours
        self.calculate_local_opinion()
        
        # calculate proportion of neighbors with same opinion
        self.calculate_opinion_similarity()
        
        # Once silent, always silent
        if self.permanently_silent:
            self.is_speaking = False
        else:
            if self.should_speak():
                self.is_speaking = True
            else:
                # First time becoming silent - mark as permanently silent
                self.is_speaking = False
                self.permanently_silent = True
    
    def calculate_local_opinion(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if neighbors:
            local_opinions = [neighbor.opinion for neighbor in neighbors]
        else:
            local_opinions = []
        local_0 = sum(1 for opinion in local_opinions if opinion == 0)
        local_1 = sum(1 for opinion in local_opinions if opinion == 1)
        """consent = 0
        for opinion in local_opinions:
            if opinion == self.opinion:
                consent +=1
        self.perceived_local_opinion = consent / len(local_opinions) if local_opinions else 0"""
        if local_0 >= local_1:
            self.perceived_local_opinion = 0
        else:
            self.perceived_local_opinion = 1
    
    def calculate_opinion_similarity(self):
        """Calculate proportion of local neighbors that share the same opinion."""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            self.opinion_similarity_ratio = 0.0
            return
        
        same_opinion_count = sum(1 for neighbor in neighbors if neighbor.opinion == self.opinion)
        self.opinion_similarity_ratio = same_opinion_count / len(neighbors)
    
    def calculate_public_opinion(self):
        media_opinion = self.model.update_media_opinion()
        if len(media_opinion) > 20:  # randomly select 10 media messages
            media_opinion = random.sample(media_opinion, 20)
        media_opinion_0 = sum(1 for opinion in media_opinion if opinion == 0)
        media_opinion_1 = sum(1 for opinion in media_opinion if opinion == 1)
        """consent = 0
        for opinion in media_opinion:
            if opinion == self.opinion:
                consent +=1
        self.perceived_public_opinion = consent / len(media_opinion) if media_opinion else 0"""
        if media_opinion_0 >= media_opinion_1:
            self.perceived_public_opinion = 0
        else:
            self.perceived_public_opinion = 1

    def should_speak(self):
        if self.perceived_public_opinion == self.opinion and self.perceived_local_opinion == self.opinion:
            return True
        elif self.confidence_level > 0.6 and self.social_isolation_fear < 0.4:
            return True
        return False

class LLMAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = random.choices([0, 1], weights=[0.6, 0.4])[0]
        self.perceived_public_opinion = 0
        self.social_isolation_fear = random.uniform(0, 1)
        self.confidence_level = random.uniform(0, 1)
        self.is_speaking = True
        self.opinion_similarity_ratio = 0.0

    def step(self):
        self.perceived_public_opinion = self.model.media_opinions[-1] if self.model.media_opinions else 0.5
        # calculate proportion of neighbors with same opinion
        self.calculate_opinion_similarity()
    
    def calculate_opinion_similarity(self):
        """Calculate proportion of local neighbors that share the same opinion."""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            self.opinion_similarity_ratio = 0.0
            return
        
        same_opinion_count = sum(1 for neighbor in neighbors if neighbor.opinion == self.opinion)
        self.opinion_similarity_ratio = same_opinion_count / len(neighbors)

class Visualization:
    @staticmethod
    def plot_silent_ratios(model):
        plt.figure(figsize=(12, 6))
        plt.plot(model.silent_ratios['opinion_0'], label='Silent Ratio Opinion 0', color='blue')
        plt.plot(model.silent_ratios['opinion_1'], label='Silent Ratio Opinion 1', color='orange')
        plt.title('Silent Ratios Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Silent Ratio')
        plt.legend()
        plt.grid()
        plt.savefig("output_plots/silent_ratios.png")
        plt.show()

    @staticmethod
    def plot_media_gap(model):
        plt.figure(figsize=(12, 6))
        plt.plot(model.media_gap_series, label='Media Gap', color='green')
        plt.title('Media Gap Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Media Gap')
        plt.legend()
        plt.grid()
        plt.savefig("output_plots/media_gap.png")
        plt.show()
    
    @staticmethod
    def plot_agent_grid(model, title="Agent Distribution on Grid"):
        plt.figure(figsize=(12, 10))
        
        # Separate agents by type and opinion
        human_opinion_0 = []
        human_opinion_1 = []
        llm_opinion_0 = []
        llm_opinion_1 = []
        
        for agent in model.schedule.agents:
            x, y = agent.pos
            if isinstance(agent, HumanAgent):
                if agent.opinion == 0:
                    human_opinion_0.append((x, y))
                else:
                    human_opinion_1.append((x, y))
            elif isinstance(agent, LLMAgent):
                if agent.opinion == 0:
                    llm_opinion_0.append((x, y))
                else:
                    llm_opinion_1.append((x, y))
        
        # Plot each group with different colors and markers
        if human_opinion_0:
            x_coords, y_coords = zip(*human_opinion_0)
            plt.scatter(x_coords, y_coords, c='blue', marker='o', s=100, 
                       label='Human - Opinion 0', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if human_opinion_1:
            x_coords, y_coords = zip(*human_opinion_1)
            plt.scatter(x_coords, y_coords, c='red', marker='o', s=100, 
                       label='Human - Opinion 1', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if llm_opinion_0:
            x_coords, y_coords = zip(*llm_opinion_0)
            plt.scatter(x_coords, y_coords, c='blue', marker='s', s=100, 
                       label='LLM - Opinion 0', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if llm_opinion_1:
            x_coords, y_coords = zip(*llm_opinion_1)
            plt.scatter(x_coords, y_coords, c='red', marker='s', s=100, 
                       label='LLM - Opinion 1', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        plt.xlim(-1, model.grid.width)
        plt.ylim(-1, model.grid.height)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output_plots/agent_grid_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    if not os.path.exists("output_plots"):
        os.makedirs("output_plots")
    
    model = SocialMedia(num_agents=400, width=20, height=20)
    for i in range(500):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    print(f"Collected model rows: {len(model_df)}")
    print(f"Collected agent rows: {len(agent_df)}")
    
    Visualization.plot_silent_ratios(model)
    Visualization.plot_media_gap(model)
    Visualization.plot_agent_grid(model, title="Agent Distribution on Grid (Final State)")