"""
Data Pipeline for Predictive Maintenance MDP
Orchestrates the complete data generation and processing workflow from 
AI4I base data through time series simulation to MDP-ready datasets.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ai4i_dataset_creator import AI4IDatasetCreator
from equipment_data_simulator import EquipmentTimeSeriesSimulator  
from maintenance_cost_simulator import MaintenanceCostSimulator

class PredictiveMaintenanceDataPipeline:
    """
    End-to-end data pipeline for predictive maintenance MDP analysis
    """
    
    def __init__(self, 
                 project_root: str = "../..",
                 random_state: int = 42):
        self.project_root = Path(project_root)
        self.random_state = random_state
        
        # Data paths
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed" 
        self.interim_dir = self.project_root / "data" / "interim"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.interim_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.file_paths = {
            'ai4i_base': self.raw_dir / "ai4i_2020_predictive_maintenance.csv",
            'timeseries': self.processed_dir / "equipment_timeseries.csv",
            'with_costs': self.processed_dir / "equipment_with_costs.csv",
            'mdp_ready': self.processed_dir / "mdp_analysis_dataset.csv",
            'summary_report': self.processed_dir / "data_pipeline_report.txt"
        }
        
    def step_1_generate_base_data(self, n_samples: int = 10000, 
                                 force_regenerate: bool = False) -> pd.DataFrame:
        """
        Generate AI4I-style base dataset
        """
        print("STEP 1: Generating AI4I Base Dataset")
        print("-" * 40)
        
        if self.file_paths['ai4i_base'].exists() and not force_regenerate:
            print(f"Loading existing dataset: {self.file_paths['ai4i_base']}")
            df = pd.read_csv(self.file_paths['ai4i_base'])
        else:
            print(f"Generating new AI4I dataset with {n_samples:,} samples...")
            creator = AI4IDatasetCreator(n_samples=n_samples, random_state=self.random_state)
            df = creator.create_dataset(str(self.file_paths['ai4i_base']))
        
        print(f"[OK] Base dataset ready: {len(df):,} records\\n")
        return df
    
    def step_2_generate_timeseries(self, ai4i_data: pd.DataFrame,
                                  n_equipment: int = 50,
                                  lifecycle_days: int = 365,
                                  measurement_interval_hours: int = 8,
                                  force_regenerate: bool = False) -> pd.DataFrame:
        """
        Convert cross-sectional data to equipment time series
        """
        print("STEP 2: Generating Equipment Time Series")
        print("-" * 40)
        
        if self.file_paths['timeseries'].exists() and not force_regenerate:
            print(f"Loading existing time series: {self.file_paths['timeseries']}")
            df = pd.read_csv(self.file_paths['timeseries'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"Simulating time series for {n_equipment} equipment units...")
            simulator = EquipmentTimeSeriesSimulator(ai4i_data, random_state=self.random_state)
            df = simulator.create_fleet_timeseries(
                n_equipment=n_equipment,
                lifecycle_days=lifecycle_days, 
                measurement_interval_hours=measurement_interval_hours
            )
            df.to_csv(self.file_paths['timeseries'], index=False)
        
        print(f"[OK] Time series data ready: {len(df):,} records\\n")
        return df
    
    def step_3_add_economic_data(self, timeseries_data: pd.DataFrame,
                                force_regenerate: bool = False) -> pd.DataFrame:
        """
        Add maintenance costs and economic metrics
        """
        print("STEP 3: Adding Economic Data")
        print("-" * 40)
        
        if self.file_paths['with_costs'].exists() and not force_regenerate:
            print(f"Loading existing cost data: {self.file_paths['with_costs']}")
            df = pd.read_csv(self.file_paths['with_costs'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("Calculating maintenance costs and economic metrics...")
            cost_simulator = MaintenanceCostSimulator(random_state=self.random_state)
            df = cost_simulator.add_economic_data(timeseries_data)
            df.to_csv(self.file_paths['with_costs'], index=False)
        
        print(f"[OK] Economic data added: {len(df):,} records\\n")
        return df
    
    def step_4_prepare_mdp_dataset(self, cost_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create analysis-ready dataset for MDP modeling
        """
        print("STEP 4: Preparing MDP Analysis Dataset") 
        print("-" * 40)
        
        # Create summary statistics per equipment and time period
        mdp_features = cost_data.groupby(['equipment_id', 'health_state']).agg({
            # State characteristics
            'health_state_name': 'first',
            'original_type': 'first',
            
            # Sensor measurements (averages)
            'tool_wear_min': 'mean',
            'air_temperature_k': 'mean', 
            'process_temperature_k': 'mean',
            'rotational_speed_rpm': 'mean',
            'torque_nm': 'mean',
            
            # Economic metrics
            'maintenance_cost': 'sum',
            'operating_cost': 'sum',
            'production_value': 'sum',
            'net_value': 'sum',
            'downtime_hours': 'sum',
            'oee': 'mean',
            
            # Counts
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'observations'}).reset_index()
        
        # Add transition information (simplified)
        mdp_features['state_duration'] = mdp_features['observations'] * 8  # hours
        mdp_features['cost_per_hour'] = mdp_features['maintenance_cost'] / mdp_features['state_duration']
        mdp_features['value_per_hour'] = mdp_features['production_value'] / mdp_features['state_duration']
        
        # Add degradation indicators
        equipment_stats = cost_data.groupby('equipment_id').agg({
            'health_state': ['min', 'max', 'mean', 'std'],
            'maintenance_cost': 'sum',
            'cumulative_net_value': 'last'
        })
        
        equipment_stats.columns = ['_'.join(col) for col in equipment_stats.columns]
        equipment_stats = equipment_stats.reset_index()
        
        # Merge equipment-level stats
        mdp_ready = mdp_features.merge(equipment_stats, on='equipment_id', how='left')
        
        # Save MDP-ready dataset
        mdp_ready.to_csv(self.file_paths['mdp_ready'], index=False)
        
        print(f"[OK] MDP dataset created: {len(mdp_ready):,} state-action pairs\\n")
        return mdp_ready
    
    def generate_summary_report(self, 
                               base_data: pd.DataFrame,
                               timeseries_data: pd.DataFrame, 
                               cost_data: pd.DataFrame,
                               mdp_data: pd.DataFrame) -> str:
        """
        Generate comprehensive pipeline summary report
        """
        print("STEP 5: Generating Summary Report")
        print("-" * 40)
        
        report_lines = [
            "PREDICTIVE MAINTENANCE DATA PIPELINE REPORT",
            "=" * 60,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Random Seed: {self.random_state}",
            "",
            "DATA PIPELINE SUMMARY",
            "-" * 30,
            f"Step 1 - Base AI4I Data: {len(base_data):,} cross-sectional records",
            f"Step 2 - Time Series: {len(timeseries_data):,} temporal observations", 
            f"Step 3 - Economic Data: {len(cost_data):,} records with costs",
            f"Step 4 - MDP Dataset: {len(mdp_data):,} state-action pairs",
            "",
            "EQUIPMENT FLEET OVERVIEW",
            "-" * 30,
        ]
        
        # Fleet statistics
        equipment_summary = timeseries_data.groupby('equipment_id').agg({
            'original_type': 'first',
            'health_state': ['min', 'max', 'mean'],
            'timestamp': ['min', 'max', 'count']
        })
        
        type_counts = timeseries_data.groupby('original_type')['equipment_id'].nunique()
        for eq_type, count in type_counts.items():
            report_lines.append(f"  {eq_type} Quality Equipment: {count} units")
        
        report_lines.extend([
            f"  Total Equipment: {timeseries_data['equipment_id'].nunique()} units",
            f"  Observation Period: {timeseries_data['timestamp'].min()} to {timeseries_data['timestamp'].max()}",
            f"  Total Operating Hours: {len(timeseries_data) * 8:,} hours",
            "",
            "HEALTH STATE DISTRIBUTION",
            "-" * 30
        ])
        
        health_dist = timeseries_data['health_state_name'].value_counts()
        for state, count in health_dist.items():
            pct = count / len(timeseries_data) * 100
            report_lines.append(f"  {state}: {count:,} ({pct:.1f}%)")
        
        # Economic summary
        if 'net_value' in cost_data.columns:
            total_production = cost_data['production_value'].sum()
            total_costs = cost_data['maintenance_cost'].sum() + cost_data['operating_cost'].sum()
            net_profit = cost_data['net_value'].sum()
            
            report_lines.extend([
                "",
                "ECONOMIC PERFORMANCE",
                "-" * 30,
                f"  Total Production Value: ${total_production:,.0f}",
                f"  Total Maintenance Costs: ${cost_data['maintenance_cost'].sum():,.0f}",
                f"  Total Operating Costs: ${cost_data['operating_cost'].sum():,.0f}",
                f"  Net Profit: ${net_profit:,.0f}",
                f"  ROI: {(net_profit / (total_costs + 1)) * 100:.1f}%"
            ])
        
        # Maintenance events
        maint_events = cost_data[cost_data['maintenance_action'] != 'None']['maintenance_action'].value_counts()
        if not maint_events.empty:
            report_lines.extend([
                "",
                "MAINTENANCE EVENTS",
                "-" * 30
            ])
            for action, count in maint_events.items():
                report_lines.append(f"  {action}: {count}")
        
        # File locations
        report_lines.extend([
            "",
            "OUTPUT FILES",
            "-" * 30,
            f"  Base Data: {self.file_paths['ai4i_base']}",
            f"  Time Series: {self.file_paths['timeseries']}",
            f"  With Costs: {self.file_paths['with_costs']}",
            f"  MDP Ready: {self.file_paths['mdp_ready']}",
            f"  This Report: {self.file_paths['summary_report']}"
        ])
        
        report_text = "\\n".join(report_lines)
        
        # Save report
        with open(self.file_paths['summary_report'], 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Summary report saved: {self.file_paths['summary_report']}\\n")
        return report_text
    
    def run_full_pipeline(self,
                         n_samples: int = 10000,
                         n_equipment: int = 50, 
                         lifecycle_days: int = 365,
                         measurement_interval_hours: int = 8,
                         force_regenerate: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data pipeline
        
        Returns:
            Dictionary containing all generated datasets
        """
        print("PREDICTIVE MAINTENANCE DATA PIPELINE")
        print("=" * 60)
        print(f"Target: {n_equipment} equipment units, {lifecycle_days} days each")
        print(f"Measurements every {measurement_interval_hours} hours\\n")
        
        # Step 1: Generate base AI4I data
        base_data = self.step_1_generate_base_data(n_samples, force_regenerate)
        
        # Step 2: Create time series
        timeseries_data = self.step_2_generate_timeseries(
            base_data, n_equipment, lifecycle_days, 
            measurement_interval_hours, force_regenerate
        )
        
        # Step 3: Add economic data
        cost_data = self.step_3_add_economic_data(timeseries_data, force_regenerate)
        
        # Step 4: Prepare MDP dataset
        mdp_data = self.step_4_prepare_mdp_dataset(cost_data)
        
        # Step 5: Generate report
        report = self.generate_summary_report(base_data, timeseries_data, cost_data, mdp_data)
        
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("All datasets are ready for MDP analysis.")
        print(f"Start with: {self.file_paths['with_costs']}")
        
        return {
            'base_data': base_data,
            'timeseries_data': timeseries_data,
            'cost_data': cost_data,
            'mdp_data': mdp_data,
            'report': report
        }

def main():
    """Run the complete data pipeline"""
    
    # Initialize pipeline
    pipeline = PredictiveMaintenanceDataPipeline()
    
    # Run with moderate-sized fleet for testing with bathtub curve
    results = pipeline.run_full_pipeline(
        n_samples=10000,      # AI4I base samples
        n_equipment=15,       # Equipment units (matching simulator)
        lifecycle_days=400,   # 13 months to see bathtub curve
        measurement_interval_hours=8,  # 3 times per day
        force_regenerate=True  # Force regenerate with bathtub curve
    )
    
    print(f"\\nDatasets generated:")
    for name, data in results.items():
        if isinstance(data, pd.DataFrame):
            print(f"  {name}: {len(data):,} records")
    
    return results

if __name__ == "__main__":
    main()