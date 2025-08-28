"""
Maintenance Cost Simulator
Adds realistic economic data to equipment time series for MDP cost optimization.
Includes maintenance costs, downtime costs, production impacts, and revenue effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MaintenanceCostSimulator:
    """
    Simulates realistic maintenance costs and economic impacts
    for manufacturing equipment predictive maintenance.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Base cost structures (in USD)
        self.base_costs = {
            'light_maintenance': {'mean': 500, 'std': 100},
            'heavy_maintenance': {'mean': 2500, 'std': 500},
            'emergency_repair': {'mean': 5000, 'std': 1000},
            'replacement': {'mean': 25000, 'std': 5000}
        }
        
        # Downtime hours by maintenance type (optimized for 98.5% availability)
        self.downtime_hours = {
            'None': 0,
            'Light Maintenance': {'mean': 0.5, 'std': 0.25},  # Quick preventive work
            'Heavy Maintenance': {'mean': 2, 'std': 0.5},     # Very efficient planned maintenance  
            'Emergency Repair': {'mean': 8, 'std': 3},        # Rapid emergency response
            'Replace': {'mean': 16, 'std': 4}                 # Efficient replacement
        }
        
        # Production parameters
        self.production_value_per_hour = {
            'L': 800,   # Low quality equipment
            'M': 1200,  # Medium quality equipment  
            'H': 1800   # High quality equipment
        }
        
        # Operating cost per hour by health state
        self.operating_costs_per_hour = {
            0: 150,  # Failed - high cost, no production
            1: 80,   # Poor - high maintenance, reduced efficiency
            2: 40,   # Fair - moderate efficiency
            3: 20,   # Good - efficient operation
            4: 15    # Excellent - optimal operation
        }
        
        # Production efficiency by health state
        self.production_efficiency = {
            0: 0.0,   # Failed - no production
            1: 0.6,   # Poor - 60% efficiency
            2: 0.8,   # Fair - 80% efficiency
            3: 0.95,  # Good - 95% efficiency
            4: 1.0    # Excellent - 100% efficiency
        }
        
    def calculate_maintenance_cost(self, maintenance_action: str, equipment_type: str = 'M') -> float:
        """
        Calculate direct maintenance cost based on action and equipment type
        """
        if maintenance_action == 'None':
            return 0.0
        
        # Map actions to cost categories
        cost_mapping = {
            'Light Maintenance': 'light_maintenance',
            'Heavy Maintenance': 'heavy_maintenance', 
            'Emergency Repair': 'emergency_repair',
            'Replace': 'replacement'
        }
        
        cost_category = cost_mapping.get(maintenance_action, 'light_maintenance')
        base_cost = self.base_costs[cost_category]
        
        # Equipment type multiplier
        type_multiplier = {'L': 0.8, 'M': 1.0, 'H': 1.3}.get(equipment_type, 1.0)
        
        # Generate cost with variation
        cost = np.random.normal(base_cost['mean'], base_cost['std']) * type_multiplier
        return max(0, cost)  # Ensure non-negative
        
    def calculate_downtime_cost(self, maintenance_action: str, equipment_type: str = 'M') -> Tuple[float, float]:
        """
        Calculate downtime hours and associated cost
        
        Returns:
            Tuple of (downtime_hours, downtime_cost)
        """
        if maintenance_action == 'None':
            return 0.0, 0.0
            
        # Get downtime hours
        if maintenance_action in self.downtime_hours:
            downtime_params = self.downtime_hours[maintenance_action]
            hours = max(0, np.random.normal(downtime_params['mean'], downtime_params['std']))
        else:
            hours = 0
            
        # Calculate lost production value
        production_rate = self.production_value_per_hour[equipment_type]
        downtime_cost = hours * production_rate
        
        return hours, downtime_cost
    
    def calculate_operating_cost(self, health_state: int, hours_operated: float = 8.0) -> float:
        """
        Calculate operating cost based on equipment health state
        """
        hourly_cost = self.operating_costs_per_hour.get(health_state, 50)
        return hourly_cost * hours_operated
    
    def calculate_production_value(self, health_state: int, equipment_type: str = 'M', hours_operated: float = 8.0) -> float:
        """
        Calculate production value based on health state and efficiency
        """
        if health_state == 0:  # Failed equipment produces nothing
            return 0.0
            
        base_production_rate = self.production_value_per_hour[equipment_type]
        efficiency = self.production_efficiency.get(health_state, 0.8)
        
        return base_production_rate * efficiency * hours_operated
    
    def add_economic_data(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive economic data to equipment time series
        """
        print("Adding economic data to time series...")
        
        # Create copy to avoid modifying original
        df = timeseries_df.copy()
        
        # Initialize cost columns
        df['maintenance_cost'] = 0.0
        df['downtime_hours'] = 0.0
        df['downtime_cost'] = 0.0
        df['operating_cost'] = 0.0
        df['production_value'] = 0.0
        df['net_value'] = 0.0
        
        # Calculate costs for each row
        for idx, row in df.iterrows():
            equipment_type = row['original_type']
            health_state = row['health_state'] 
            maintenance_action = row['maintenance_action']
            
            # Maintenance costs
            maint_cost = self.calculate_maintenance_cost(maintenance_action, equipment_type)
            df.at[idx, 'maintenance_cost'] = maint_cost
            
            # Downtime costs
            downtime_hrs, downtime_cost = self.calculate_downtime_cost(maintenance_action, equipment_type)
            df.at[idx, 'downtime_hours'] = downtime_hrs
            df.at[idx, 'downtime_cost'] = downtime_cost
            
            # Operating costs (assuming 8-hour measurement periods)
            operating_hours = 8.0 - downtime_hrs  # Actual operating hours
            operating_cost = self.calculate_operating_cost(health_state, max(0, operating_hours))
            df.at[idx, 'operating_cost'] = operating_cost
            
            # Production value
            production_value = self.calculate_production_value(health_state, equipment_type, max(0, operating_hours))
            df.at[idx, 'production_value'] = production_value
            
            # Net value (production minus all costs)
            total_costs = maint_cost + downtime_cost + operating_cost
            net_value = production_value - total_costs
            df.at[idx, 'net_value'] = net_value
        
        # Add cumulative metrics per equipment
        df['cumulative_maintenance_cost'] = df.groupby('equipment_id')['maintenance_cost'].cumsum()
        df['cumulative_production_value'] = df.groupby('equipment_id')['production_value'].cumsum()
        df['cumulative_net_value'] = df.groupby('equipment_id')['net_value'].cumsum()
        
        # Calculate additional business metrics
        df = self._add_business_metrics(df)
        
        return df
    
    def _add_business_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced business metrics and KPIs
        """
        # Overall Equipment Effectiveness (OEE) components
        df['availability'] = 1 - (df['downtime_hours'] / 8.0)  # 8-hour periods
        df['performance'] = df['production_efficiency'] = [
            self.production_efficiency.get(state, 0) for state in df['health_state']
        ]
        df['quality_rate'] = np.where(df['original_type'] == 'H', 0.98, 
                            np.where(df['original_type'] == 'M', 0.95, 0.92))
        
        # OEE calculation
        df['oee'] = df['availability'] * df['performance'] * df['quality_rate']
        
        # Maintenance efficiency metrics
        df['mtbf_indicator'] = df.groupby('equipment_id')['maintenance_action'].apply(
            lambda x: (x != 'None').cumsum()
        ).values  # Maintenance event counter
        
        # Cost ratios
        df['maintenance_cost_ratio'] = df['maintenance_cost'] / (df['production_value'] + 0.01)  # Avoid division by zero
        df['total_cost_ratio'] = (df['maintenance_cost'] + df['operating_cost']) / (df['production_value'] + 0.01)
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive cost summary report
        """
        total_records = len(df)
        
        # Overall financial metrics
        total_maintenance_cost = df['maintenance_cost'].sum()
        total_downtime_cost = df['downtime_cost'].sum()
        total_operating_cost = df['operating_cost'].sum()
        total_production_value = df['production_value'].sum()
        net_profit = df['net_value'].sum()
        
        # Equipment type breakdown
        type_summary = df.groupby('original_type').agg({
            'maintenance_cost': 'sum',
            'production_value': 'sum', 
            'net_value': 'sum',
            'oee': 'mean',
            'equipment_id': 'nunique'
        }).round(2)
        
        # Maintenance action costs
        maintenance_summary = df[df['maintenance_action'] != 'None'].groupby('maintenance_action').agg({
            'maintenance_cost': ['count', 'mean', 'sum'],
            'downtime_hours': 'mean',
            'downtime_cost': 'sum'
        }).round(2)
        
        # Health state analysis
        health_summary = df.groupby('health_state_name').agg({
            'operating_cost': 'mean',
            'production_value': 'mean',
            'oee': 'mean'
        }).round(3)
        
        report = {
            'total_records': total_records,
            'financial_summary': {
                'total_maintenance_cost': round(total_maintenance_cost, 2),
                'total_downtime_cost': round(total_downtime_cost, 2),
                'total_operating_cost': round(total_operating_cost, 2),
                'total_production_value': round(total_production_value, 2),
                'net_profit': round(net_profit, 2),
                'roi': round((net_profit / (total_maintenance_cost + total_operating_cost + 0.01)) * 100, 2)
            },
            'equipment_type_summary': type_summary.to_dict(),
            'maintenance_summary': maintenance_summary,
            'health_state_summary': health_summary.to_dict()
        }
        
        return report

def main():
    """Add economic data to equipment time series"""
    
    # Load time series data
    ts_path = "../../data/processed/equipment_timeseries.csv"
    timeseries_df = pd.read_csv(ts_path)
    timeseries_df['timestamp'] = pd.to_datetime(timeseries_df['timestamp'])
    
    print(f"Loaded time series data: {len(timeseries_df):,} records")
    
    # Create cost simulator
    cost_simulator = MaintenanceCostSimulator()
    
    # Add economic data
    enhanced_df = cost_simulator.add_economic_data(timeseries_df)
    
    # Generate summary report
    report = cost_simulator.generate_summary_report(enhanced_df)
    
    # Print summary
    print("\n" + "="*50)
    print("ECONOMIC ANALYSIS SUMMARY")
    print("="*50)
    
    financial = report['financial_summary']
    print(f"Total Production Value: ${financial['total_production_value']:,.0f}")
    print(f"Total Maintenance Cost: ${financial['total_maintenance_cost']:,.0f}")
    print(f"Total Operating Cost: ${financial['total_operating_cost']:,.0f}")
    print(f"Net Profit: ${financial['net_profit']:,.0f}")
    print(f"ROI: {financial['roi']:.1f}%")
    
    print(f"\nEquipment Count by Type:")
    type_counts = enhanced_df.groupby('original_type')['equipment_id'].nunique()
    for eq_type, count in type_counts.items():
        print(f"  {eq_type} Quality: {count} units")
    
    print(f"\nMaintenance Events:")
    maint_events = enhanced_df[enhanced_df['maintenance_action'] != 'None']['maintenance_action'].value_counts()
    for action, count in maint_events.items():
        print(f"  {action}: {count}")
    
    # Save enhanced dataset
    output_path = "../../data/processed/equipment_with_costs.csv"
    enhanced_df.to_csv(output_path, index=False)
    print(f"\nEnhanced dataset saved to: {output_path}")
    
    return enhanced_df, report

if __name__ == "__main__":
    main()