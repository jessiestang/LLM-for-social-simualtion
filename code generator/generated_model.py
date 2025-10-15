
import random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class AgentType(Agent):
    """An agent with a unique ID and type."""
    
    def __init__(self, unique_id, model, agent_type):
        super().__init__(unique_id, model)
        self.agent_type = agent_type  # Type of the agent (0 or 1)
        self.happiness_threshold = 0.5  # Happiness threshold for staying

    def assess_neighbors(self):
        """Assess the types of neighboring agents to determine happiness."""
        similar_neighbors = 0
        total_neighbors = 0
        
        # Get the list of neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        for neighbor in neighbors:
            total_neighbors += 1
            if neighbor.agent_type == self.agent_type:
                similar_neighbors += 1
        
        # Calculate the proportion of similar-type neighbors
        if total_neighbors > 0:
            return similar_neighbors / total_neighbors
        return 0

    def step(self):
        """Decide whether to stay or move based on neighbor assessment."""
        happiness = self.assess_neighbors()
        
        if happiness >= self.happiness_threshold:
            # Stay in the current position
            return
        else:
            # Move to a random neighboring position
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            if neighbors:
                new_position = random.choice(neighbors).pos
                self.model.grid.move_agent(self, new_position)

class LatticeModel(Model):
    """A model with a number of agents on a lattice."""
    
    def __init__(self, width, height, N):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Create agents
        for i in range(self.num_agents):
            agent_type = random.choice([0, 1])  # Randomly assign agent type
            agent = AgentType(i, self, agent_type)
            self.schedule.add(agent)
            
            # Place agent on a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Type": "agent_type"}
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

# Example of running the model
if __name__ == "__main__":
    model = LatticeModel(10, 10, 100)  # Create a model with a 10x10 grid and 100 agents
    for i in range(100):  # Run the model for 100 steps
        model.step()
    # You can add code here to analyze the results from the model
    agent_data = model.datacollector.get_agent_vars_dataframe()
    agent_counts = agent_data["Type"].value_counts()
    print(agent_counts)
