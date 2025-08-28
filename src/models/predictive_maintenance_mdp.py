"""
Predictive Maintenance MDP Model
A clean, data-driven Markov Decision Process for equipment maintenance optimization.

Refactored from manufacturing_mdp_project.py to work with real equipment data
and provide modular, extensible maintenance decision support.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MaintenanceAction:
    """Define maintenance actions with costs and effectiveness"""
    name: str
    cost: float
    effectiveness: float  # How much it improves equipment state (0-1)
    downtime_hours: float = 0.0

class PredictiveMaintenanceMDP:
    """
    Data-driven Markov Decision Process for Equipment Maintenance
    
    States: Equipment health levels (0=Failed, 1=Poor, 2=Fair, 3=Good, 4=Excellent)
    Actions: Maintenance interventions with varying costs and effectiveness
    Objective: Minimize long-term costs while maximizing availability
    """
    
    def __init__(self, cost_parameters: Dict = None):
        # Health states (0=Failed, 4=Excellent)
        self.n_states = 5
        self.states = list(range(self.n_states))
        self.state_names = {
            0: 'Failed',
            1: 'Poor', 
            2: 'Fair',
            3: 'Good',
            4: 'Excellent'
        }
        
        # Default cost parameters (final tuning for 98.5% uptime policy)
        default_costs = {
            'light_maintenance': 750,     # Increased to reduce over-maintenance
            'heavy_maintenance': 5000,    # Further increased to make "Do Nothing" more attractive
            'emergency_repair': 5000,
            'replacement': 25000,
            'production_value_per_hour': 1000,
            'downtime_cost_multiplier': 1.0
        }
        
        self.cost_params = {**default_costs, **(cost_parameters or {})}
        
        # Define maintenance actions (optimized downtime for 98.5% availability)
        self.actions = {
            0: MaintenanceAction("Do Nothing", cost=0, effectiveness=0, downtime_hours=0),
            1: MaintenanceAction("Light Maintenance", 
                                cost=self.cost_params['light_maintenance'], 
                                effectiveness=0.3, 
                                downtime_hours=0.5),
            2: MaintenanceAction("Heavy Maintenance", 
                                cost=self.cost_params['heavy_maintenance'], 
                                effectiveness=0.7, 
                                downtime_hours=2),
            3: MaintenanceAction("Emergency Repair", 
                                cost=self.cost_params['emergency_repair'], 
                                effectiveness=0.5, 
                                downtime_hours=8),
            4: MaintenanceAction("Replace", 
                                cost=self.cost_params['replacement'], 
                                effectiveness=1.0, 
                                downtime_hours=16)
        }
        
        # Operating efficiency by state (affects production)
        self.production_efficiency = {
            0: 0.0,   # Failed - no production
            1: 0.6,   # Poor - 60% efficiency  
            2: 0.8,   # Fair - 80% efficiency
            3: 0.95,  # Good - 95% efficiency
            4: 1.0    # Excellent - 100% efficiency
        }
        
        # Operating costs per time period by state
        self.operating_costs = {
            0: 200,   # Failed - high maintenance needs
            1: 80,    # Poor - frequent issues
            2: 40,    # Fair - moderate issues  
            3: 20,    # Good - minimal issues
            4: 15     # Excellent - optimal operation
        }
    
    def update_from_data(self, equipment_data: pd.DataFrame):
        """
        Update MDP parameters based on real equipment data
        
        Args:
            equipment_data: DataFrame with columns including health_state, 
                          maintenance_cost, production_value, etc.
        """
        print("Updating MDP parameters from equipment data...")
        
        # Update maintenance costs from actual data
        if 'maintenance_cost' in equipment_data.columns:
            maint_data = equipment_data[equipment_data['maintenance_cost'] > 0]
            
            for action_name in ['Light Maintenance', 'Heavy Maintenance', 'Emergency Repair']:
                action_costs = maint_data[maint_data['maintenance_action'] == action_name]['maintenance_cost']
                if not action_costs.empty:
                    avg_cost = action_costs.mean()
                    # Find and update corresponding action
                    for action_id, action in self.actions.items():
                        if action.name == action_name:
                            action.cost = avg_cost
                            print(f"Updated {action_name} cost to ${avg_cost:.0f}")
        
        # Update operating costs from data
        if 'operating_cost' in equipment_data.columns:
            state_costs = equipment_data.groupby('health_state')['operating_cost'].mean()
            for state, cost in state_costs.items():
                if state in self.operating_costs:
                    self.operating_costs[state] = cost
                    print(f"Updated operating cost for {self.state_names[state]}: ${cost:.0f}")
        
        # Update production value
        if 'production_value' in equipment_data.columns:
            avg_production = equipment_data['production_value'].mean() / 8  # Per hour
            self.cost_params['production_value_per_hour'] = avg_production
            print(f"Updated production value: ${avg_production:.0f}/hour")
    
    def get_transition_matrix(self, action: int, 
                            transition_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Calculate transition probabilities for given maintenance action
        
        Args:
            action: Action index
            transition_data: Optional real data for learning transitions
        """
        P = np.zeros((self.n_states, self.n_states))
        action_obj = self.actions[action]
        
        # If real transition data is provided, learn from it
        if transition_data is not None:
            return self._learn_transitions_from_data(action, transition_data)
        
        # Otherwise use model-based transitions
        for i in range(self.n_states):
            if i == 0:  # Failed state
                if action in [3, 4]:  # Repair or Replace
                    if action == 4:  # Replace
                        P[i, 4] = 1.0  # New equipment -> Excellent
                    else:  # Emergency repair
                        P[i, 2] = 0.8  # Usually -> Fair
                        P[i, 1] = 0.2  # Sometimes -> Poor
                else:
                    P[i, 0] = 1.0  # Stay failed without intervention
            else:
                # Base degradation probabilities 
                base_degrade = 0.25 - 0.04 * i  # Higher states degrade slower
                
                # Maintenance effect
                improvement = action_obj.effectiveness
                
                if action == 4:  # Replace
                    P[i, 4] = 1.0
                else:
                    # Probability of improvement
                    improve_prob = min(0.6, improvement * 0.8)
                    
                    # Degradation probability (reduced by maintenance)
                    degrade_prob = max(0.05, base_degrade * (1 - improvement * 0.5))
                    
                    # Stay same probability  
                    stay_prob = 1 - improve_prob - degrade_prob
                    
                    # Assign probabilities
                    if improve_prob > 0 and i < self.n_states - 1:
                        # Can improve up to 2 states based on effectiveness
                        improve_states = min(2, int(improvement * 3))
                        target_state = min(self.n_states - 1, i + improve_states)
                        P[i, target_state] = improve_prob
                    
                    if degrade_prob > 0 and i > 0:
                        P[i, i-1] = degrade_prob
                        
                    P[i, i] = max(0, stay_prob)
                    
                    # Normalize probabilities
                    row_sum = P[i, :].sum()
                    if row_sum > 0:
                        P[i, :] = P[i, :] / row_sum
        
        return P
    
    def _learn_transitions_from_data(self, action: int, data: pd.DataFrame) -> np.ndarray:
        """Learn transition probabilities from historical data"""
        P = np.zeros((self.n_states, self.n_states))
        
        # Filter data for specific maintenance action
        action_name = self.actions[action].name
        if action_name == "Do Nothing":
            action_data = data[data['maintenance_action'] == 'None']
        else:
            action_data = data[data['maintenance_action'] == action_name]
        
        if action_data.empty:
            # Fallback to model-based if no data
            return self.get_transition_matrix(action, None)
        
        # Count transitions
        for i in range(self.n_states):
            current_state_data = action_data[action_data['health_state'] == i]
            
            if current_state_data.empty:
                # Use uniform distribution or model-based fallback
                P[i, :] = 1.0 / self.n_states
                continue
            
            # Find next states (simplified - in practice would need temporal ordering)
            next_states = current_state_data['health_state'].values
            
            for j in range(self.n_states):
                count = np.sum(next_states == j)
                P[i, j] = count / len(next_states) if len(next_states) > 0 else 0
            
            # Normalize
            row_sum = P[i, :].sum()
            if row_sum > 0:
                P[i, :] = P[i, :] / row_sum
            else:
                P[i, i] = 1.0  # Stay in same state if no data
        
        return P
    
    def calculate_immediate_cost(self, state: int, action: int, 
                               time_period_hours: float = 8.0) -> float:
        """
        Calculate immediate cost for state-action pair
        
        Args:
            state: Current health state
            action: Maintenance action
            time_period_hours: Length of time period
        """
        action_obj = self.actions[action]
        
        # Direct maintenance cost
        maintenance_cost = action_obj.cost
        
        # Operating cost based on current state
        operating_cost = self.operating_costs.get(state, 50) * time_period_hours
        
        # Production loss due to downtime
        downtime_cost = (action_obj.downtime_hours * 
                        self.cost_params['production_value_per_hour'] * 
                        self.cost_params['downtime_cost_multiplier'])
        
        # Production loss due to reduced efficiency
        efficiency = self.production_efficiency.get(state, 0.8)
        production_loss = ((1 - efficiency) * 
                          (time_period_hours - action_obj.downtime_hours) * 
                          self.cost_params['production_value_per_hour'])
        
        total_cost = maintenance_cost + operating_cost + downtime_cost + production_loss
        return total_cost
    
    def value_iteration(self, discount_factor: float = 0.95, 
                       tolerance: float = 1e-6, 
                       max_iterations: int = 1000,
                       transition_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MDP using value iteration
        
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
                    P = self.get_transition_matrix(action, transition_data)
                    
                    # Bellman equation (cost minimization)
                    expected_future_cost = discount_factor * np.sum(P[state, :] * old_values)
                    total_cost = immediate_cost + expected_future_cost
                    action_values.append(total_cost)
                
                # Choose action with minimum expected cost
                best_action = np.argmin(action_values)
                values[state] = action_values[best_action]
                policy[state] = best_action
            
            # Check convergence
            if np.max(np.abs(values - old_values)) < tolerance:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        return values, policy
    
    def analyze_policy(self, policy: np.ndarray) -> pd.DataFrame:
        """Create comprehensive policy analysis"""
        
        analysis_data = []
        
        for state in range(self.n_states):
            action_idx = int(policy[state])
            action = self.actions[action_idx]
            
            analysis_data.append({
                'State': self.state_names[state],
                'State_Index': state,
                'Optimal_Action': action.name,
                'Action_Cost': action.cost,
                'Downtime_Hours': action.downtime_hours,
                'Operating_Cost_Per_Period': self.operating_costs[state],
                'Production_Efficiency': self.production_efficiency[state],
                'Total_Period_Cost': self.calculate_immediate_cost(state, action_idx)
            })
        
        return pd.DataFrame(analysis_data)
    
    def simulate_policy(self, policy: np.ndarray, 
                       initial_state: int = 4,
                       time_periods: int = 1000,
                       n_simulations: int = 100,
                       transition_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Simulate policy performance over time
        
        Returns:
            Dictionary with simulation results and metrics
        """
        total_costs = []
        availability_rates = []
        maintenance_frequencies = []
        
        for sim in range(n_simulations):
            state = initial_state
            costs = []
            downtime_hours = 0
            maintenance_events = 0
            
            for t in range(time_periods):
                action_idx = int(policy[state])
                action = self.actions[action_idx]
                
                # Record costs and downtime
                cost = self.calculate_immediate_cost(state, action_idx)
                costs.append(cost)
                downtime_hours += action.downtime_hours
                
                if action.name != "Do Nothing":
                    maintenance_events += 1
                
                # Transition to next state
                P = self.get_transition_matrix(action_idx, transition_data)
                state = np.random.choice(self.states, p=P[state, :])
            
            total_costs.append(np.sum(costs))
            availability_rates.append(1 - (downtime_hours / (time_periods * 8)))  # Assume 8-hour periods
            maintenance_frequencies.append(maintenance_events / time_periods)
        
        return {
            'total_costs': total_costs,
            'average_cost': np.mean(total_costs),
            'cost_std': np.std(total_costs),
            'average_availability': np.mean(availability_rates),
            'average_maintenance_frequency': np.mean(maintenance_frequencies),
            'simulation_summary': {
                'total_simulations': n_simulations,
                'time_periods': time_periods,
                'cost_per_period': np.mean(total_costs) / time_periods
            }
        }

def load_equipment_data(file_path: str) -> pd.DataFrame:
    """Helper function to load and preprocess equipment data"""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        print(f"Data file not found: {file_path}")
        return pd.DataFrame()

def main():
    """Demonstrate MDP with real equipment data"""
    
    print("Predictive Maintenance MDP Analysis")
    print("=" * 60)
    
    # Load equipment data if available
    data_path = "../../data/processed/equipment_with_costs.csv"
    equipment_data = load_equipment_data(data_path)
    
    # Initialize MDP
    mdp = PredictiveMaintenanceMDP()
    
    # Update parameters from real data if available
    if not equipment_data.empty:
        mdp.update_from_data(equipment_data)
        print(f"\nUsing real data: {len(equipment_data):,} records")
    else:
        print("\nUsing default parameters (no data file found)")
    
    # Solve MDP
    print("\nSolving MDP using Value Iteration...")
    transition_data = equipment_data if not equipment_data.empty else None
    values, policy = mdp.value_iteration(transition_data=transition_data)
    
    # Analyze results
    print("\nOptimal Maintenance Policy:")
    policy_df = mdp.analyze_policy(policy)
    print(policy_df.to_string(index=False))
    
    print(f"\nOptimal Value Function (Expected Costs):")
    for i, (state_name, value) in enumerate(zip(mdp.state_names.values(), values)):
        print(f"   {state_name:10}: ${value:8.0f}")
    
    # Simulate policy
    print(f"\nSimulating Policy Performance...")
    results = mdp.simulate_policy(policy, time_periods=365, n_simulations=100, 
                                 transition_data=transition_data)
    
    print(f"   Average annual cost: ${results['average_cost']:,.0f}")
    print(f"   Daily cost: ${results['simulation_summary']['cost_per_period']:.0f}")
    print(f"   Equipment availability: {results['average_availability']*100:.1f}%")
    print(f"   Maintenance frequency: {results['average_maintenance_frequency']*100:.1f}% of periods")
    
    return mdp, values, policy, results

if __name__ == "__main__":
    main()