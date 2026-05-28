import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from mesa import Model
sys.path.append(os.path.abspath( "../code_generator"))
from generated_shelling_model import SegregationModel, Visualization  # Assuming the model code is in segregation_model.py

def run_simulation(num_agents, width, height, num_runs):
    segregation_indices = []

    for run in range(num_runs):
        model = SegregationModel(num_agents=num_agents, width=width, height=height)
        
        for _ in range(300):  # Run for 300 time steps
            model.step()
        
        # Calculate segregation index
        unhappy_agents = sum(1 for agent in model.schedule.agents if agent.decision_signal < agent.homogeneity_preference)
        segregation_index = unhappy_agents / num_agents
        segregation_indices.append(segregation_index)

    return segregation_indices

def analyze_results(segregation_indices):
    mean_segregation = np.mean(segregation_indices)
    std_segregation = np.std(segregation_indices)

    return mean_segregation, std_segregation

def plot_results(segregation_indices):
    plt.figure(figsize=(10, 5))
    plt.hist(segregation_indices, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Segregation Indices")
    plt.xlabel("Segregation Index")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(segregation_indices), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.legend()
    plt.savefig('output_plots/segregation_index_distribution.png')
    plt.show()

def main():
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')

    num_agents = 300
    width = 20
    height = 20
    num_runs = 50

    segregation_indices = run_simulation(num_agents, width, height, num_runs)
    mean_segregation, std_segregation = analyze_results(segregation_indices)

    print(f"Mean Segregation Index: {mean_segregation:.4f}")
    print(f"Standard Deviation of Segregation Index: {std_segregation:.4f}")

    plot_results(segregation_indices)

if __name__ == "__main__":
    main()