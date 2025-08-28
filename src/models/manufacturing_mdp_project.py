"""
Manufacturing Equipment Predictive Maintenance MDP
Building on Markov Chain expertise for decision science applications

Problem: Optimize maintenance decisions for critical manufacturing equipment
- Equipment degrades through discrete health states
- Maintenance actions affect transition probabilities and costs
- Goal: Minimize long-term costs while avoiding failures

This extends your drug dosing project framework to manufacturing decision science.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MaintenanceAction:
    """Define maintenance actions with costs and effectiveness"""
    name: str
    cost: float
    effectiveness: float  # How much it improves equipment state

class EquipmentMaintenanceMDP:
    """
    Markov Decision Process for Equipment Maintenance
    
    States: Equipment health levels (0=Failed, 1=Poor, 2=Fair, 3=Good, 4=Excellent)
    Actions: Do Nothing, Light Maintenance, Heavy Maintenance, Replace
    
    Similar structure to your drug dosing project but for manufacturing decisions
    """
    
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.states = list(range(n_states))  # 0=Failed, 4=Excellent
        
        # Define maintenance actions (similar to your dosing levels)
        self.actions = {
            0: MaintenanceAction("Do Nothing", cost=0, effectiveness=0),
            1: MaintenanceAction("Light Maintenance", cost=500, effectiveness=0.3),
            2: MaintenanceAction("Heavy Maintenance", cost=2000, effectiveness=0.7),
            3: MaintenanceAction("Replace", cost=15000, effectiveness=1.0)
        }
        
        # Operating costs per state (higher for worse conditions)
        self.operating_costs = np.array([10000, 3000, 1000, 200, 100])  # Per time period
        
        # Production loss multipliers (failed equipment = no production)
        self.production_loss = np.array([1.0, 0.4, 0.2, 0.1, 0.0])  # Fraction of lost production
        self.production_value_per_period = 5000
        
    def get_transition_matrix(self, action: int) -> np.ndarray:
        """
        Calculate transition probabilities for given maintenance action
        Similar to your drug concentration transitions, but for equipment degradation
        """
        P = np.zeros((self.n_states, self.n_states))
        action_obj = self.actions[action]
        
        for i in range(self.n_states):
            if i == 0:  # Failed state
                if action == 3:  # Replace
                    P[i, 4] = 1.0  # New equipment
                else:
                    P[i, 0] = 1.0  # Stay failed
            else:
                # Base degradation probabilities (without maintenance)
                base_degrade_prob = 0.3 - 0.05 * i  # Higher states degrade slower
                
                # Maintenance reduces degradation and can improve state
                improvement = action_obj.effectiveness
                
                if action == 3:  # Replace
                    P[i, 4] = 1.0
                else:
                    # Probability of improvement (up to 2 states)
                    improve_prob = min(0.8, improvement)
                    new_state = min(self.n_states - 1, i + int(improvement * 2))
                    
                    # Degradation probability reduced by maintenance
                    degrade_prob = max(0.05, base_degrade_prob * (1 - improvement))
                    
                    # Stay same probability
                    stay_prob = 1 - improve_prob - degrade_prob
                    
                    # Assign probabilities
                    if new_state > i:
                        P[i, new_state] = improve_prob
                    if i > 0:
                        P[i, i-1] = degrade_prob
                    P[i, i] = max(0, stay_prob)
                    
                    # Normalize to ensure valid probabilities
                    P[i, :] = P[i, :] / P[i, :].sum()
        
        return P
    
    def calculate_immediate_cost(self, state: int, action: int) -> float:
        """
        Calculate immediate cost for state-action pair
        Extends your cost modeling to manufacturing context
        """
        maintenance_cost = self.actions[action].cost
        operating_cost = self.operating_costs[state]
        production_loss_cost = self.production_loss[state] * self.production_value_per_period
        
        return maintenance_cost + operating_cost + production_loss_cost
    
    def value_iteration(self, discount_factor: float = 0.95, tolerance: float = 1e-6, 
                       max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MDP using value iteration (similar to your steady-state calculations)
        
        Returns:
            values: Optimal value function
            policy: Optimal policy (action for each state)
        """
        values = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        
        for iteration in range(max_iterations):
            old_values = values.copy()
            
            for state in range(self.n_states):
                action_values = []
                
                for action in self.actions.keys():
                    immediate_cost = self.calculate_immediate_cost(state, action)
                    P = self.get_transition_matrix(action)
                    
                    # Bellman equation (minimizing cost, so negative of standard formulation)
                    expected_future_value = discount_factor * np.sum(P[state, :] * old_values)
                    total_value = immediate_cost + expected_future_value
                    action_values.append(total_value)
                
                # Choose action with minimum cost
                best_action = np.argmin(action_values)
                values[state] = action_values[best_action]
                policy[state] = best_action
            
            # Check convergence
            if np.max(np.abs(values - old_values)) < tolerance:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        return values, policy
    
    def simulate_policy(self, policy: np.ndarray, initial_state: int = 4, 
                       time_periods: int = 1000, n_simulations: int = 100) -> Dict:
        """
        Simulate the optimal policy to evaluate performance
        Similar to your simulation validation approach
        """
        total_costs = []
        state_histories = []
        
        for sim in range(n_simulations):
            state = initial_state
            costs = []
            states = [state]
            
            for t in range(time_periods):
                action = int(policy[state])
                cost = self.calculate_immediate_cost(state, action)
                costs.append(cost)
                
                # Transition to next state
                P = self.get_transition_matrix(action)
                state = np.random.choice(self.states, p=P[state, :])
                states.append(state)
            
            total_costs.append(np.sum(costs))
            state_histories.append(states)
        
        return {
            'total_costs': total_costs,
            'average_cost': np.mean(total_costs),
            'cost_std': np.std(total_costs),
            'state_histories': state_histories
        }
    
    def analyze_policy(self, policy: np.ndarray) -> pd.DataFrame:
        """Create policy analysis similar to your results tables"""
        policy_df = pd.DataFrame({
            'State': ['Failed', 'Poor', 'Fair', 'Good', 'Excellent'],
            'State_Index': range(self.n_states),
            'Optimal_Action': [self.actions[int(policy[i])].name for i in range(self.n_states)],
            'Action_Cost': [self.actions[int(policy[i])].cost for i in range(self.n_states)],
            'Operating_Cost': self.operating_costs,
            'Production_Loss': self.production_loss * self.production_value_per_period
        })
        return policy_df

def main():
    """
    Main analysis - extends your project structure to manufacturing
    """
    print("Manufacturing Equipment Maintenance MDP Analysis")
    print("=" * 60)
    
    # Initialize MDP
    mdp = EquipmentMaintenanceMDP()
    
    # Solve using value iteration
    print("\n1. Solving MDP using Value Iteration...")
    values, policy = mdp.value_iteration()
    
    # Display optimal policy
    print("\n2. Optimal Maintenance Policy:")
    policy_df = mdp.analyze_policy(policy)
    print(policy_df.to_string(index=False))
    
    print(f"\n3. Optimal Value Function:")
    for i, (state_name, value) in enumerate(zip(['Failed', 'Poor', 'Fair', 'Good', 'Excellent'], values)):
        print(f"   {state_name:10}: ${value:8.0f} expected cost")
    
    # Simulate policy performance
    print("\n4. Simulating Policy Performance...")
    results = mdp.simulate_policy(policy, initial_state=4, time_periods=500, n_simulations=200)
    
    print(f"   Average total cost over 500 periods: ${results['average_cost']:,.0f}")
    print(f"   Standard deviation: ${results['cost_std']:,.0f}")
    print(f"   Cost per period: ${results['average_cost']/500:.0f}")
    
    # Create visualizations
    create_visualizations(mdp, policy, values, results)

def create_visualizations(mdp, policy, values, results):
    """Create visualizations similar to your project plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Transition matrices for each action
    ax1 = axes[0, 0]
    action_to_plot = 1  # Light maintenance
    P = mdp.get_transition_matrix(action_to_plot)
    sns.heatmap(P, annot=True, fmt='.3f', cmap='Blues', ax=ax1,
                xticklabels=['Failed', 'Poor', 'Fair', 'Good', 'Excellent'],
                yticklabels=['Failed', 'Poor', 'Fair', 'Good', 'Excellent'])
    ax1.set_title(f'Transition Matrix: {mdp.actions[action_to_plot].name}')
    ax1.set_xlabel('Next State')
    ax1.set_ylabel('Current State')
    
    # Plot 2: Optimal policy
    ax2 = axes[0, 1]
    states = ['Failed', 'Poor', 'Fair', 'Good', 'Excellent']
    actions = [mdp.actions[int(policy[i])].name for i in range(mdp.n_states)]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    bars = ax2.bar(states, range(len(states)), color=colors)
    ax2.set_title('Optimal Maintenance Policy by State')
    ax2.set_ylabel('State Index')
    
    # Add action labels on bars
    for i, (bar, action) in enumerate(zip(bars, actions)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                action, ha='center', va='center', rotation=90, fontweight='bold')
    
    # Plot 3: Value function
    ax3 = axes[1, 0]
    ax3.plot(states, values, 'bo-', linewidth=2, markersize=8)
    ax3.set_title('Optimal Value Function (Expected Costs)')
    ax3.set_ylabel('Expected Cost ($)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost distribution from simulation
    ax4 = axes[1, 1]
    ax4.hist(results['total_costs'], bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(results['average_cost'], color='red', linestyle='--', 
                label=f'Mean: ${results["average_cost"]:,.0f}')
    ax4.set_title('Distribution of Total Costs (500 periods)')
    ax4.set_xlabel('Total Cost ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
