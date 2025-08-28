"""
Equipment Health State Classifier
Bridges sensor data and MDP health states using machine learning classification.
Maps continuous sensor readings to discrete health states for MDP decision making.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

class EquipmentHealthClassifier:
    """
    Machine learning classifier to predict equipment health states
    from sensor measurements for MDP state estimation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Model components
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        
        # Feature names for interpretability
        self.feature_names = [
            'tool_wear_min',
            'air_temperature_k', 
            'process_temperature_k',
            'rotational_speed_rpm',
            'torque_nm',
            'temp_difference',
            'power_calculated'
        ]
        
        self.target_name = 'health_state'
        self.state_names = {
            0: 'Failed',
            1: 'Poor',
            2: 'Fair', 
            3: 'Good',
            4: 'Excellent'
        }
        
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw sensor data
        """
        features_df = df.copy()
        
        # Basic sensor readings
        feature_cols = []
        
        # Tool wear (key degradation indicator)
        if 'tool_wear_min' in df.columns:
            feature_cols.append('tool_wear_min')
        
        # Temperature measurements
        if 'air_temperature_k' in df.columns:
            feature_cols.append('air_temperature_k')
        if 'process_temperature_k' in df.columns:
            feature_cols.append('process_temperature_k')
            
        # Mechanical measurements  
        if 'rotational_speed_rpm' in df.columns:
            feature_cols.append('rotational_speed_rpm')
        if 'torque_nm' in df.columns:
            feature_cols.append('torque_nm')
        
        # Derived features
        if 'air_temperature_k' in df.columns and 'process_temperature_k' in df.columns:
            features_df['temp_difference'] = (df['process_temperature_k'] - 
                                            df['air_temperature_k'])
            feature_cols.append('temp_difference')
        
        if 'rotational_speed_rpm' in df.columns and 'torque_nm' in df.columns:
            features_df['power_calculated'] = (df['torque_nm'] * 
                                             df['rotational_speed_rpm'] * 
                                             2 * np.pi / 60)
            feature_cols.append('power_calculated')
        
        # Keep only engineered features
        self.feature_names = feature_cols
        return features_df[feature_cols + (['health_state'] if 'health_state' in df.columns else [])]
    
    def train(self, training_data: pd.DataFrame, 
              test_size: float = 0.2,
              validation_splits: int = 5) -> Dict:
        """
        Train the health state classifier
        
        Args:
            training_data: DataFrame with sensor data and health_state labels
            test_size: Fraction of data for testing
            validation_splits: Number of CV folds
            
        Returns:
            Dictionary with training results and metrics
        """
        print("Training Equipment Health Classifier")
        print("-" * 40)
        
        # Prepare features
        feature_data = self.prepare_features(training_data)
        
        if 'health_state' not in feature_data.columns:
            raise ValueError("Training data must include 'health_state' column")
        
        # Separate features and target
        X = feature_data[self.feature_names].fillna(0)
        y = feature_data['health_state']
        
        print(f"Training samples: {len(X):,}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {sorted(y.unique())}")
        
        # Check class distribution for stratification
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        # Use stratification only if all classes have >= 2 samples
        use_stratify = all(count >= 2 for count in class_counts)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if use_stratify else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_test_pred = self.classifier.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_train_scaled, y_train, 
            cv=validation_splits, scoring='accuracy'
        )
        
        # Training metrics
        train_accuracy = (y_train_pred == y_train).mean()
        test_accuracy = (y_test_pred == y_test).mean()
        
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")
        print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'test_predictions': y_test_pred,
            'test_actual': y_test
        }
    
    def predict_health_state(self, sensor_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict health states from sensor measurements
        
        Args:
            sensor_data: DataFrame or array with sensor measurements
            
        Returns:
            Array of predicted health states (0-4)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Handle DataFrame input
        if isinstance(sensor_data, pd.DataFrame):
            feature_data = self.prepare_features(sensor_data)
            X = feature_data[self.feature_names].fillna(0)
        else:
            X = sensor_data
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)
        
        return predictions
    
    def predict_health_probability(self, sensor_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict health state probabilities
        
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Handle DataFrame input
        if isinstance(sensor_data, pd.DataFrame):
            feature_data = self.prepare_features(sensor_data)
            X = feature_data[self.feature_names].fillna(0)
        else:
            X = sensor_data
        
        # Scale and predict probabilities
        X_scaled = self.scaler.transform(X)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return probabilities
    
    def create_diagnostic_plots(self, training_results: Dict, save_path: str = None):
        """
        Create diagnostic plots for model performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Feature importance
        importance_df = training_results['feature_importance']
        axes[0, 0].barh(importance_df['feature'], importance_df['importance'])
        axes[0, 0].set_title('Feature Importance')
        axes[0, 0].set_xlabel('Importance')
        
        # Plot 2: Confusion matrix
        cm = training_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], 
                   xticklabels=list(self.state_names.values()),
                   yticklabels=list(self.state_names.values()))
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Plot 3: Cross-validation scores
        cv_scores = training_results['cv_scores']
        axes[1, 0].boxplot(cv_scores)
        axes[1, 0].set_title('Cross-Validation Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticklabels(['5-Fold CV'])
        
        # Plot 4: Prediction distribution
        test_pred = training_results['test_predictions']
        test_actual = training_results['test_actual']
        
        pred_counts = pd.Series(test_pred).value_counts().sort_index()
        actual_counts = pd.Series(test_actual).value_counts().sort_index()
        
        x = np.arange(len(self.state_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, [actual_counts.get(i, 0) for i in range(5)], 
                      width, label='Actual', alpha=0.8)
        axes[1, 1].bar(x + width/2, [pred_counts.get(i, 0) for i in range(5)], 
                      width, label='Predicted', alpha=0.8)
        
        axes[1, 1].set_title('Test Set: Actual vs Predicted Distribution')
        axes[1, 1].set_xlabel('Health State')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(list(self.state_names.values()))
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostic plots saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, file_path: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'state_names': self.state_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """Load trained model from disk"""
        model_data = joblib.load(file_path)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.state_names = model_data['state_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {file_path}")

def main():
    """Train and evaluate equipment health classifier"""
    
    print("EQUIPMENT HEALTH STATE CLASSIFIER")
    print("=" * 50)
    
    # Load equipment data
    data_path = "../../data/processed/equipment_with_costs.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded equipment data: {len(df):,} records")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Run the data pipeline first to generate training data")
        return None
    
    # Initialize classifier
    classifier = EquipmentHealthClassifier()
    
    # Train model
    training_results = classifier.train(df, test_size=0.3)
    
    # Create diagnostic plots
    plot_path = "../../reports/figures/health_classifier_diagnostics.png"
    classifier.create_diagnostic_plots(training_results, plot_path)
    
    # Save trained model
    model_path = "../../models/equipment_health_classifier.pkl"
    classifier.save_model(model_path)
    
    # Print classification report
    print("\\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(training_results['classification_report'])
    
    # Example prediction
    print("\\nExample Predictions:")
    print("-" * 20)
    sample_data = df.sample(5)[classifier.feature_names]
    predictions = classifier.predict_health_state(sample_data)
    probabilities = classifier.predict_health_probability(sample_data)
    
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        pred_state = predictions[i]
        pred_name = classifier.state_names[pred_state]
        confidence = probabilities[i][pred_state]
        
        print(f"Sample {i+1}: Predicted {pred_name} (confidence: {confidence:.2f})")
        print(f"  Tool wear: {row['tool_wear_min']:.1f} min")
        print(f"  Temperature diff: {row.get('temp_difference', 0):.1f} K")
    
    return classifier, training_results

if __name__ == "__main__":
    main()