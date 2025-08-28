"""
AI4I 2020 Dataset Creator
Creates a sample dataset matching the AI4I 2020 Predictive Maintenance Dataset structure
for development and testing purposes.

Based on: S. Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance Applications'
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class AI4IDatasetCreator:
    """
    Creates synthetic data matching AI4I 2020 dataset structure
    """
    
    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_base_features(self) -> pd.DataFrame:
        """Generate base features following AI4I dataset characteristics"""
        
        # Product quality variants (L, M, H with different proportions)
        product_quality = np.random.choice(['L', 'M', 'H'], 
                                         size=self.n_samples, 
                                         p=[0.6, 0.3, 0.1])  # L=Low quality more common
        
        # Air temperature [K] - normal around 300K (27°C)
        air_temp = np.random.normal(300, 2, self.n_samples)
        
        # Process temperature [K] - correlated with air temp but higher
        process_temp = air_temp + np.random.normal(10, 1, self.n_samples)
        
        # Rotational speed [rpm] - varies by quality
        base_speed = {'L': 1550, 'M': 1500, 'H': 1450}
        rotational_speed = np.array([
            np.random.normal(base_speed[q], 100) for q in product_quality
        ])
        
        # Torque [Nm] - inversely related to speed
        base_torque = 40
        torque = base_torque - (rotational_speed - 1500) * 0.01 + np.random.normal(0, 5, self.n_samples)
        torque = np.maximum(torque, 10)  # Minimum torque
        
        # Tool wear [min] - cumulative, higher for low quality
        quality_multiplier = {'L': 1.5, 'M': 1.0, 'H': 0.7}
        tool_wear = np.array([
            np.random.gamma(2, quality_multiplier[q] * 50) for q in product_quality
        ])
        tool_wear = np.minimum(tool_wear, 250)  # Maximum tool wear
        
        df = pd.DataFrame({
            'UDI': range(1, self.n_samples + 1),
            'Product ID': [f"{q}{i:05d}" for i, q in enumerate(product_quality, 1)],
            'Type': product_quality,
            'Air temperature [K]': np.round(air_temp, 1),
            'Process temperature [K]': np.round(process_temp, 1),
            'Rotational speed [rpm]': np.round(rotational_speed, 0).astype(int),
            'Torque [Nm]': np.round(torque, 1),
            'Tool wear [min]': np.round(tool_wear, 0).astype(int)
        })
        
        return df
    
    def calculate_failure_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate failure modes based on feature values"""
        
        # Tool Wear Failure (TWF) - when tool wear exceeds threshold
        twf = (df['Tool wear [min]'] > 200).astype(int)
        
        # Heat Dissipation Failure (HDF) - high temp difference and low speed
        temp_diff = df['Process temperature [K]'] - df['Air temperature [K]']
        hdf = ((temp_diff > 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
        
        # Power Failure (PWF) - extreme power conditions (more realistic thresholds)
        power = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60  # Power calculation
        pwf = ((power > 8000) | (power < 1500)).astype(int)
        
        # Overstrain Failure (OSF) - high tool wear and high torque
        osf = ((df['Tool wear [min]'] > 180) & (df['Torque [Nm]'] > 60)).astype(int)
        
        # Random Failures (RNF) - 0.1% random failure rate
        rnf = (np.random.random(len(df)) < 0.001).astype(int)
        
        # Machine Failure - any failure mode triggered
        machine_failure = (twf | hdf | pwf | osf | rnf).astype(int)
        
        df['TWF'] = twf
        df['HDF'] = hdf
        df['PWF'] = pwf
        df['OSF'] = osf
        df['RNF'] = rnf
        df['Machine failure'] = machine_failure
        
        return df
    
    def create_dataset(self, save_path: str = None) -> pd.DataFrame:
        """Create complete AI4I-style dataset"""
        
        print("Generating AI4I 2020-style dataset...")
        df = self.generate_base_features()
        df = self.calculate_failure_modes(df)
        
        # Reorder columns to match AI4I format
        columns_order = [
            'UDI', 'Product ID', 'Type', 
            'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
        ]
        df = df[columns_order]
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to: {save_path}")
        
        # Print summary statistics
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(df):,}")
        print(f"Machine failures: {df['Machine failure'].sum():,} ({df['Machine failure'].mean()*100:.1f}%)")
        print(f"Quality distribution: {df['Type'].value_counts().to_dict()}")
        
        failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for mode in failure_modes:
            count = df[mode].sum()
            print(f"{mode}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df

def main():
    """Generate sample AI4I dataset"""
    creator = AI4IDatasetCreator(n_samples=10000)
    
    # Create dataset and save
    save_path = "../../data/raw/ai4i_2020_predictive_maintenance.csv"
    dataset = creator.create_dataset(save_path)
    
    return dataset

if __name__ == "__main__":
    main()