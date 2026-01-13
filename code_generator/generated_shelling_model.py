import numpy as np
import matplotlib.pyplot as plt
import random
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class AgentRacialType1(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.homogeneity_preference = random.uniform(0, 1)
        self.decision_signal = 0  # Initialize decision_signal

    def decide_movement(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        same_race_count = sum(1 for neighbor in neighbors if isinstance(neighbor, AgentRacialType1))
        vacant_count = sum(1 for neighbor in neighbors if neighbor is None)
        
        racial_composition_neighbor = same_race_count / len(neighbors) if len(neighbors) > 0 else 0
        vacant_spot_availability = vacant_count / len(neighbors) if len(neighbors) > 0 else 0
        
        self.decision_signal = (0.8 * racial_composition_neighbor) + (0.2 * vacant_spot_availability)  # Store decision_signal
        
        if self.decision_signal < self.homogeneity_preference:
            self.move()

    def move(self):
        # Get all empty cells in the grid
        vacant_spots = []
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                if self.model.grid.is_cell_empty((x, y)):
                    vacant_spots.append((x, y))
        if vacant_spots:
            new_position = random.choice(vacant_spots)
            self.model.grid.move_agent(self, new_position)

class AgentRacialType2(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.homogeneity_preference = random.uniform(0, 1)
        self.decision_signal = 0  # Initialize decision_signal

    def decide_movement(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        same_race_count = sum(1 for neighbor in neighbors if isinstance(neighbor, AgentRacialType2))
        vacant_count = sum(1 for neighbor in neighbors if neighbor is None)
        
        racial_composition_neighbor = same_race_count / len(neighbors) if len(neighbors) > 0 else 0
        vacant_spot_availability = vacant_count / len(neighbors) if len(neighbors) > 0 else 0
        
        self.decision_signal = (0.8 * racial_composition_neighbor) + (0.2 * vacant_spot_availability)  # Store decision_signal
        
        if self.decision_signal < self.homogeneity_preference:
            self.move()

    def move(self):
        # Get all empty cells in the grid
        vacant_spots = []
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                if self.model.grid.is_cell_empty((x, y)):
                    vacant_spots.append((x, y))
        if vacant_spots:
            new_position = random.choice(vacant_spots)
            self.model.grid.move_agent(self, new_position)

class SegregationModel(Model):
    def __init__(self, num_agents, width, height):
        super().__init__()  # Explicitly initialize the Model
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.unhappy_agents_count = []
        
        for i in range(self.num_agents // 2):
            agent = AgentRacialType1(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        for i in range(self.num_agents // 2, self.num_agents):
            agent = AgentRacialType2(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Unhappy": "decision_signal"}
        )

    def step(self):
        self.datacollector.collect(self)
        unhappy_agents = 0
        for agent in self.schedule.agents:
            agent.decide_movement()
            if agent.decision_signal < agent.homogeneity_preference:
                unhappy_agents += 1
        self.unhappy_agents_count.append(unhappy_agents)
        self.schedule.step()

class Visualization:
    @staticmethod
    def plot_grid(model, title):
        # Create RGB image: gray=vacant, red=Type1, blue=Type2
        grid_data = np.zeros((model.grid.height, model.grid.width, 3))
        # Initialize with gray for vacant cells
        grid_data[:, :] = [0.8, 0.8, 0.8]  # Gray
        
        for agent in model.schedule.agents:
            x, y = agent.pos
            if isinstance(agent, AgentRacialType1):
                grid_data[y, x] = [1, 0, 0]  # Red for Type 1
            elif isinstance(agent, AgentRacialType2):
                grid_data[y, x] = [0, 0, 1]  # Blue for Type 2
        
        plt.imshow(grid_data)
        plt.title(title)
        plt.axis('off')
        plt.savefig(f'output_plots/{title}.png')
        plt.show()

    @staticmethod
    def plot_unhappy_agents(unhappy_agents_count):
        plt.plot(unhappy_agents_count)
        plt.title("Number of Unhappy Agents Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Unhappy Agents")
        plt.savefig('output_plots/unhappy_agents_over_time.png')
        plt.show()

def main():
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')
    
    model = SegregationModel(num_agents=300, width=20, height=20)
    
    # Initial State
    Visualization.plot_grid(model, "Initial State")
    
    for i in range(300):
        model.step()
    
    # Midpoint
    Visualization.plot_grid(model, "Midpoint")
    
    for i in range(300):
        model.step()
    
    # Final State
    Visualization.plot_grid(model, "Final State")
    
    # Plot unhappy agents over time
    Visualization.plot_unhappy_agents(model.unhappy_agents_count)

if __name__ == "__main__":
    main()