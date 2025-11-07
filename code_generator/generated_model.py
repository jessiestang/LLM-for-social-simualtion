import random
import matplotlib.pyplot as plt
import numpy as np
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class EconomicAgent(Agent):
    """An agent with economic status and decision-making capabilities."""
    
    def __init__(self, unique_id, model, agent_type, economic_status, income_level):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.economic_status = economic_status
        self.income_level = income_level
        self.satisfaction = None

    def move(self):
        """Move to a new location if conditions are met."""
        # Get neighboring POSITIONS (not agents)
        neighbor_positions = self.model.grid.get_neighborhood(
            self.pos, 
            moore=True, 
            include_center=False
        )
        
        best_location = None
        best_income = -1
        
        for neighbor_pos in neighbor_positions:
            avg_income = self.model.get_average_income(neighbor_pos)
            job_availability = self.model.get_job_availability(neighbor_pos)
            
            if (avg_income > self.income_level and 
                job_availability > 0 and 
                self.satisfaction < self.model.tolerance_threshold):
                if avg_income > best_income:
                    best_income = avg_income
                    best_location = neighbor_pos
        
        if best_location is not None:
            self.model.grid.move_agent(self, best_location)

    def assess_neighborhood(self):
        """Assess the neighborhood composition and economic status."""
        # Get neighboring AGENTS
        neighbors = self.model.grid.get_neighbors(
            self.pos, 
            moore=True, 
            include_center=False
        )
        
        satisfied_count = sum(1 for neighbor in neighbors if neighbor.economic_status == self.economic_status)
        total_neighbors = len(neighbors)
        
        self.satisfaction = satisfied_count / total_neighbors if total_neighbors > 0 else 0

    def step(self):
        """Agent's step in the model."""
        self.assess_neighborhood()
        self.move()


class EconomicModel(Model):
    """A model with some number of agents."""
    
    def __init__(self, num_agents, width, height):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.data_collector = DataCollector(agent_reporters={"Satisfaction": "satisfaction"})
        
        # Create agents
        for i in range(self.num_agents):
            agent_type = 'A' if i < self.num_agents / 2 else 'B'
            economic_status = random.choice(['low', 'medium', 'high'])
            income_level = random.randint(1000, 5000)
            agent = EconomicAgent(i, self, agent_type, economic_status, income_level)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.tolerance_threshold = 0.5
        self.datacollector = DataCollector(
            agent_reporters={"Satisfaction": "satisfaction"}
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

    def get_average_income(self, pos):
        """Calculate average income in a given cell."""
        agents = self.grid.get_cell_list_contents([pos])
        if len(agents) == 0:
            return 0
        return np.mean([agent.income_level for agent in agents])

    def get_job_availability(self, pos):
        """Dummy function for job availability."""
        return random.randint(0, 5)  # Random job availability for simplicity


class Visualization:
    """Class for visualizing the model's results."""
    
    @staticmethod
    def plot_satisfaction(data):
        """Plot the satisfaction levels of agents over time."""
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(data)), data, color='blue')
        plt.xlabel('Time Steps')
        plt.ylabel('Average Satisfaction Level')
        plt.title('Satisfaction Levels Over Time')
        plt.savefig("output_plots/satisfaction_levels.png")
        plt.show()

    @staticmethod
    def plot_agents(model):
        """Plot the positions of agents on a grid."""
        grid = np.zeros((model.grid.width, model.grid.height))
        for agent in model.schedule.agents:
            if agent.satisfaction is not None:
                if agent.satisfaction >= model.tolerance_threshold:
                    grid[agent.pos[0], agent.pos[1]] = 1 if agent.agent_type == 'A' else 2
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='coolwarm', origin='upper')
        plt.colorbar(ticks=[0, 1, 2], label='Agent Type (0: None, 1: A, 2: B)')
        plt.title('Agent Positions on Grid')
        plt.savefig("output_plots/agent_positions.png")
        plt.show()


if __name__ == "__main__":
    if not os.path.exists("output_plots"):
        os.makedirs("output_plots")
    
    model = EconomicModel(num_agents=100, width=10, height=10)
    satisfaction_data = []
    
    for i in range(10):
        model.step()
        avg_satisfaction = np.mean([agent.satisfaction for agent in model.schedule.agents if agent.satisfaction is not None])
        satisfaction_data.append(avg_satisfaction)
    
    Visualization.plot_satisfaction(satisfaction_data)
    Visualization.plot_agents(model)