import numpy as np
from model import SpiralOfSilenceModel  # Assuming the model is defined in a file named model.py

def evaluate_sensitivity_of_beta():
    beta_values = np.arange(0.1, 1.0, 0.1)
    results = {}

    for beta in beta_values:
        speaking_proportions = []
        
        for _ in range(50):  # Number of iterations
            model = SpiralOfSilenceModel(num_human_agents=50, num_llm_agents=50, width=10, height=10,
                                          beta=beta, gamma=0.5, influence_threshold=0.5, isolation_threshold=0.7)
            for _ in range(100):  # Number of time steps
                model.step()
            
            # Collect the proportion of speaking agents
            speaking_count = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.is_speaking)
            total_human_agents = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent))
            speaking_proportion = speaking_count / total_human_agents if total_human_agents > 0 else 0
            speaking_proportions.append(speaking_proportion)
        
        # Calculate the average proportion of speaking agents for this beta
        results[beta] = np.mean(speaking_proportions)

    return results

if __name__ == "__main__":
    output = evaluate_sensitivity_of_beta()
    for beta, avg_proportion in output.items():
        print(f"Beta: {beta:.1f}, Average Proportion of Speaking Agents: {avg_proportion:.2f}")