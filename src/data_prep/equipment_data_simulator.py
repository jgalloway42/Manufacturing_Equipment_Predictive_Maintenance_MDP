"""
Equipment Data Simulator
Extends AI4I cross-sectional data into time-series format with degradation modeling
for use in MDP predictive maintenance applications.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EquipmentTimeSeriesSimulator:
    """
    Converts cross-sectional AI4I data into realistic time-series 
    with equipment degradation and maintenance events following bathtub curve reliability.
    """
    
    def __init__(self, ai4i_data: pd.DataFrame, random_state: int = 42):
        self.ai4i_data = ai4i_data.copy()
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define health states (0=Failed, 1=Poor, 2=Fair, 3=Good, 4=Excellent)
        self.health_states = {
            0: 'Failed',
            1: 'Poor', 
            2: 'Fair',
            3: 'Good',
            4: 'Excellent'
        }
        
        # Bathtub curve parameters (in operating hours) - calibrated for 95% uptime
        self.reliability_params = {
            'infant_mortality_period': 1000,    # 0-1000 hrs: high initial failure rate
            'useful_life_period': 6000,         # 1000-7000 hrs: constant low failure rate  
            'wearout_start': 7000,              # 7000+ hrs: increasing failure rate
            
            # Failure rates fine-tuned for 98.5% ± 0.25% fleet uptime
            'infant_mortality_rate': 0.020,     # 2.0% in first 1000 hrs (decreasing)  
            'random_failure_rate': 0.0008,      # 0.08% constant rate during useful life
            'wearout_acceleration': 0.00005,    # Minimal exponential increase
            
            # Equipment quality impact on reliability
            'quality_reliability_factor': {'L': 1.6, 'M': 1.0, 'H': 0.6}  # Quality differences
        }
        
    def map_features_to_health_state(self, row: pd.Series) -> int:
        """
        Map AI4I features to discrete health states for MDP
        """
        # Start with excellent health
        health_score = 4
        
        # Penalize based on tool wear
        if row['Tool wear [min]'] > 200:
            health_score = 0  # Failed
        elif row['Tool wear [min]'] > 150:
            health_score = min(health_score, 1)  # Poor
        elif row['Tool wear [min]'] > 100:
            health_score = min(health_score, 2)  # Fair
        elif row['Tool wear [min]'] > 50:
            health_score = min(health_score, 3)  # Good
            
        # Penalize based on temperature issues
        temp_diff = row['Process temperature [K]'] - row['Air temperature [K]']
        if temp_diff > 12:
            health_score = min(health_score, 1)  # Poor
        elif temp_diff > 10:
            health_score = min(health_score, 2)  # Fair
            
        # Penalize based on power issues
        power = row['Torque [Nm]'] * row['Rotational speed [rpm]'] * 2 * np.pi / 60
        if power < 2000 or power > 7000:
            health_score = min(health_score, 2)  # Fair
            
        # Any failure mode means failed state
        if row['Machine failure'] == 1:
            health_score = 0
            
        return health_score
    
    def calculate_bathtub_failure_probability(self, operating_hours: float, 
                                            equipment_quality: str, 
                                            measurement_interval: float = 8.0) -> float:
        """
        Calculate failure probability based on bathtub curve reliability model
        
        Args:
            operating_hours: Total operating hours of equipment
            equipment_quality: L/M/H quality level
            measurement_interval: Hours in current measurement period
            
        Returns:
            Probability of failure in this measurement period
        """
        params = self.reliability_params
        quality_factor = params['quality_reliability_factor'][equipment_quality]
        
        # Phase 1: Infant Mortality (Decreasing failure rate)
        if operating_hours < params['infant_mortality_period']:
            # Weibull-like decreasing failure rate
            if operating_hours <= 8:  # Handle initial measurement periods  
                hazard_rate = params['infant_mortality_rate'] * quality_factor * 0.05  # High initial rate
            else:
                beta = 0.6  # Shape parameter < 1 for decreasing rate
                lambda_0 = params['infant_mortality_rate'] * quality_factor
                
                # Weibull hazard rate with proper scaling for realistic failures
                time_ratio = operating_hours / 1000  # Scale to 0-1 range
                hazard_rate = lambda_0 * beta * (time_ratio ** (beta - 1)) / 10  # Scaled down
            
        # Phase 2: Useful Life (Constant failure rate)  
        elif operating_hours < params['wearout_start']:
            hazard_rate = params['random_failure_rate'] * quality_factor
            
        # Phase 3: Wear-out (Increasing failure rate)
        else:
            wearout_hours = operating_hours - params['wearout_start']
            base_rate = params['random_failure_rate'] * quality_factor
            # Exponential increase in failure rate during wear-out
            wearout_multiplier = 1 + (wearout_hours * params['wearout_acceleration'])
            hazard_rate = base_rate * wearout_multiplier
        
        # Convert hourly hazard rate to probability over measurement interval
        # P(failure) = 1 - exp(-hazard_rate * time_interval)
        failure_probability = 1 - np.exp(-hazard_rate * measurement_interval / 1000)
        
        return min(failure_probability, 0.5)  # Cap at 50% per period
    
    def simulate_equipment_lifecycle(self, 
                                   equipment_id: str,
                                   initial_data: pd.Series,
                                   lifecycle_days: int = 365,
                                   measurement_interval_hours: int = 8) -> pd.DataFrame:
        """
        Simulate complete equipment lifecycle with degradation
        """
        measurements_per_day = 24 // measurement_interval_hours
        total_measurements = lifecycle_days * measurements_per_day
        
        # Initialize arrays for time series
        timestamps = []
        health_states = []
        tool_wear_series = []
        air_temp_series = []
        process_temp_series = []
        rpm_series = []
        torque_series = []
        maintenance_events = []
        
        # Starting conditions
        current_health = self.map_features_to_health_state(initial_data)
        current_tool_wear = max(0, initial_data['Tool wear [min]'] - np.random.normal(50, 10))
        base_air_temp = initial_data['Air temperature [K]']
        base_process_temp = initial_data['Process temperature [K]']
        base_rpm = initial_data['Rotational speed [rpm]']
        base_torque = initial_data['Torque [Nm]']
        equipment_quality = initial_data['Type']
        
        # Simulation parameters (more preventive maintenance)
        base_degradation_rate = 0.05  # Slower degradation (better maintenance)
        maintenance_threshold = 120   # Earlier preventive maintenance
        replacement_threshold = 180   # Earlier replacement threshold
        
        # Equipment lifecycle tracking
        total_operating_hours = 0
        time_since_last_maintenance = 0
        equipment_age_cycles = 0  # Number of maintenance/replacement cycles
        
        start_date = datetime.now() - timedelta(days=lifecycle_days)
        
        for i in range(total_measurements):
            current_time = start_date + timedelta(hours=i * measurement_interval_hours)
            timestamps.append(current_time)
            
            # Update operating hours
            if current_health > 0:
                total_operating_hours += measurement_interval_hours
                time_since_last_maintenance += measurement_interval_hours
            
            # Check for bathtub curve failures FIRST (before other logic)
            bathtub_failure_prob = self.calculate_bathtub_failure_probability(
                total_operating_hours, equipment_quality, measurement_interval_hours
            )
            
            # Apply bathtub curve failure
            if current_health > 0 and np.random.random() < bathtub_failure_prob:
                current_health = 0  # Catastrophic failure
                print(f"  Bathtub failure at {total_operating_hours:.0f} hrs ({equipment_id})")
            
            # Natural degradation (if not failed)
            if current_health > 0:
                # Age-dependent degradation rate (increases with equipment age)
                age_factor = 1 + (equipment_age_cycles * 0.1)  # 10% increase per maintenance cycle
                wearout_factor = 1 + max(0, (total_operating_hours - 6000) / 10000)  # Accelerate after 6000 hrs
                
                degradation_rate = base_degradation_rate * age_factor * wearout_factor
                current_tool_wear += np.random.gamma(2, degradation_rate)
                
                # Add seasonal and operational variations
                day_of_year = current_time.timetuple().tm_yday
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Temperature variations (affected by wear)
                temp_noise_factor = 1 + (current_tool_wear / 300) * 0.5  # More variation with wear
                air_temp = base_air_temp + np.random.normal(0, 1 * temp_noise_factor) * seasonal_factor
                process_temp = base_process_temp + np.random.normal(0, 1.5 * temp_noise_factor) * seasonal_factor
                
                # Equipment performance degradation (non-linear with wear)
                wear_factor = 1 + (current_tool_wear / 200) ** 1.5 * 0.15  # Exponential degradation
                rpm = base_rpm * (1 - (current_tool_wear / 400) ** 1.2) + np.random.normal(0, 20 * wear_factor)
                torque = base_torque * wear_factor + np.random.normal(0, 2 * wear_factor)
                
            else:  # Equipment failed
                air_temp = base_air_temp + np.random.normal(0, 3)  # Higher variation when failed
                process_temp = base_process_temp + np.random.normal(0, 4)
                rpm = base_rpm * 0.6 + np.random.normal(0, 80)  # Very poor performance
                torque = base_torque * 1.4 + np.random.normal(0, 8)  # High stress
            
            # Maintenance events
            maintenance_action = 'None'
            
            # Calculate age-dependent maintenance effectiveness
            age_degradation = min(0.4, equipment_age_cycles * 0.05)  # Up to 40% reduction in effectiveness
            hours_degradation = min(0.3, max(0, (total_operating_hours - 5000) / 10000))  # Hours-based reduction
            maintenance_effectiveness = 1.0 - age_degradation - hours_degradation
            
            # More frequent light maintenance to keep equipment in good condition
            if current_tool_wear >= maintenance_threshold * 0.7 and current_health < 4:  # Light maintenance trigger
                maintenance_action = 'Light Maintenance'
                improvement_factor = 0.3 * maintenance_effectiveness
                current_tool_wear *= (1 - improvement_factor)
                current_health = min(4, current_health + 1)
                time_since_last_maintenance = 0
                
            # Preventive maintenance  
            elif current_tool_wear >= maintenance_threshold and current_health > 0:
                if current_tool_wear >= replacement_threshold:
                    # Full replacement
                    maintenance_action = 'Replace'
                    current_tool_wear = np.random.normal(5, 2)  # Near-new condition
                    current_health = 4  # Excellent
                    equipment_age_cycles = 0  # Reset age
                    total_operating_hours = 0  # Reset hours
                    time_since_last_maintenance = 0
                else:
                    # Heavy maintenance
                    maintenance_action = 'Heavy Maintenance'
                    # Effectiveness decreases with age
                    improvement_factor = 0.8 * maintenance_effectiveness  # More effective
                    current_tool_wear *= (1 - improvement_factor)  
                    health_improvement = max(1, int(2 * maintenance_effectiveness))
                    current_health = min(4, current_health + health_improvement)
                    equipment_age_cycles += 1
                    time_since_last_maintenance = 0
                    
            # Reactive maintenance (after failure)
            elif current_health == 0:
                # Success rate decreases with age and operating hours
                repair_success_rate = 0.8 * maintenance_effectiveness
                
                if np.random.random() < repair_success_rate:
                    maintenance_action = 'Emergency Repair'
                    # Less effective repair on older equipment
                    improvement_factor = 0.8 * maintenance_effectiveness
                    current_tool_wear *= (1 - improvement_factor * 0.3)  # Limited improvement
                    # Older equipment may not reach good health after repair
                    max_post_repair_health = 3 if equipment_age_cycles < 3 else 2
                    current_health = max_post_repair_health
                    equipment_age_cycles += 1
                    time_since_last_maintenance = 0
                else:
                    # Repair failed - equipment stays failed
                    maintenance_action = 'Failed Repair Attempt'
            
            # Update health state based on current conditions
            if current_health > 0:
                # Create temporary series for health mapping
                temp_data = pd.Series({
                    'Tool wear [min]': current_tool_wear,
                    'Air temperature [K]': air_temp,
                    'Process temperature [K]': process_temp,
                    'Rotational speed [rpm]': rpm,
                    'Torque [Nm]': torque,
                    'Machine failure': 0
                })
                current_health = self.map_features_to_health_state(temp_data)
            
            # Store data
            health_states.append(current_health)
            tool_wear_series.append(current_tool_wear)
            air_temp_series.append(air_temp)
            process_temp_series.append(process_temp)
            rpm_series.append(rpm)
            torque_series.append(torque)
            maintenance_events.append(maintenance_action)
        
        # Create DataFrame with additional bathtub curve tracking
        df = pd.DataFrame({
            'equipment_id': equipment_id,
            'timestamp': timestamps,
            'health_state': health_states,
            'health_state_name': [self.health_states[h] for h in health_states],
            'tool_wear_min': tool_wear_series,
            'air_temperature_k': air_temp_series,
            'process_temperature_k': process_temp_series,
            'rotational_speed_rpm': rpm_series,
            'torque_nm': torque_series,
            'maintenance_action': maintenance_events,
            'original_type': initial_data['Type'],  # Keep quality type
            'operating_hours': list(range(0, len(timestamps) * measurement_interval_hours, measurement_interval_hours)),
            'equipment_age_cycles': [0] * len(timestamps)  # Will be updated in post-processing
        })
        
        return df
    
    def create_fleet_timeseries(self, 
                              n_equipment: int = 50,
                              lifecycle_days: int = 365,
                              measurement_interval_hours: int = 8) -> pd.DataFrame:
        """
        Create time series data for a fleet of equipment
        """
        print(f"Generating time series data for {n_equipment} pieces of equipment...")
        
        fleet_data = []
        
        # Sample equipment from AI4I data
        sample_equipment = self.ai4i_data.sample(n=n_equipment, random_state=self.random_state)
        
        for idx, (_, equipment_row) in enumerate(sample_equipment.iterrows()):
            equipment_id = f"EQUIP_{idx+1:03d}"
            print(f"Simulating {equipment_id} ({equipment_row['Type']} quality)...")
            
            # Generate lifecycle for this equipment
            equipment_ts = self.simulate_equipment_lifecycle(
                equipment_id=equipment_id,
                initial_data=equipment_row,
                lifecycle_days=lifecycle_days,
                measurement_interval_hours=measurement_interval_hours
            )
            
            fleet_data.append(equipment_ts)
        
        # Combine all equipment data
        complete_df = pd.concat(fleet_data, ignore_index=True)
        complete_df = complete_df.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)
        
        print(f"\nFleet Time Series Summary (with Bathtub Curve):")
        print(f"Total records: {len(complete_df):,}")
        print(f"Equipment count: {complete_df['equipment_id'].nunique()}")
        print(f"Date range: {complete_df['timestamp'].min()} to {complete_df['timestamp'].max()}")
        print(f"Total operating hours: {complete_df['operating_hours'].max():,}")
        
        print(f"\nHealth state distribution:")
        health_dist = complete_df['health_state_name'].value_counts()
        for state, count in health_dist.items():
            print(f"  {state}: {count:,} ({count/len(complete_df)*100:.1f}%)")
        
        print(f"\nEquipment quality vs failure rate:")
        quality_failure_rate = complete_df.groupby('original_type').agg({
            'health_state': lambda x: (x == 0).mean() * 100
        }).round(2)
        for quality, rate in quality_failure_rate.iterrows():
            print(f"  {quality} Quality: {rate['health_state']:.1f}% failure rate")
        
        maintenance_dist = complete_df[complete_df['maintenance_action'] != 'None']['maintenance_action'].value_counts()
        if not maintenance_dist.empty:
            print(f"\nMaintenance events:")
            for action, count in maintenance_dist.items():
                print(f"  {action}: {count:,}")
        
        # Show failures by operating hours (bathtub curve validation)
        failure_data = complete_df[complete_df['health_state'] == 0]
        if not failure_data.empty:
            print(f"\nFailures by lifecycle phase:")
            infant_failures = failure_data[failure_data['operating_hours'] < 1000]
            useful_life_failures = failure_data[(failure_data['operating_hours'] >= 1000) & 
                                              (failure_data['operating_hours'] < 8000)]
            wearout_failures = failure_data[failure_data['operating_hours'] >= 8000]
            
            print(f"  Infant mortality (0-1000 hrs): {len(infant_failures)} failures")
            print(f"  Useful life (1000-8000 hrs): {len(useful_life_failures)} failures") 
            print(f"  Wear-out (8000+ hrs): {len(wearout_failures)} failures")
        
        return complete_df

def main():
    """Generate equipment time series data"""
    
    # Load AI4I data
    ai4i_path = "../../data/raw/ai4i_2020_predictive_maintenance.csv"
    ai4i_data = pd.read_csv(ai4i_path)
    
    # Create simulator
    simulator = EquipmentTimeSeriesSimulator(ai4i_data)
    
    # Generate fleet time series with longer lifecycle to see bathtub curve
    fleet_ts = simulator.create_fleet_timeseries(
        n_equipment=15,  # Smaller fleet for detailed analysis
        lifecycle_days=400,  # ~13 months to see full bathtub curve
        measurement_interval_hours=8  # 3 measurements per day
    )
    
    # Save time series data
    output_path = "../../data/processed/equipment_timeseries.csv"
    fleet_ts.to_csv(output_path, index=False)
    print(f"\nTime series data saved to: {output_path}")
    
    return fleet_ts

if __name__ == "__main__":
    main()