"""
IoT Sensor Data Simulator for Predictive Maintenance
Generates realistic sensor data with failure patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import random


class IoTSensorSimulator:
    """Simulates IoT sensor data for industrial equipment"""
    
    def __init__(self, equipment_types: List[str] = None):
        self.equipment_types = equipment_types or [
            'pump', 'motor', 'compressor', 'turbine', 'generator'
        ]
        
        # Define sensor parameters for each equipment type
        self.sensor_configs = {
            'pump': {
                'temperature': {'normal': (60, 80), 'failure': (90, 120)},
                'vibration': {'normal': (0.1, 0.3), 'failure': (0.8, 1.5)},
                'pressure': {'normal': (2.0, 3.0), 'failure': (0.5, 1.5)},
                'flow_rate': {'normal': (100, 150), 'failure': (50, 90)},
                'power_consumption': {'normal': (5, 8), 'failure': (10, 15)}
            },
            'motor': {
                'temperature': {'normal': (70, 90), 'failure': (110, 140)},
                'vibration': {'normal': (0.05, 0.2), 'failure': (0.6, 1.2)},
                'current': {'normal': (10, 15), 'failure': (20, 30)},
                'rpm': {'normal': (1750, 1800), 'failure': (1600, 1700)},
                'power_consumption': {'normal': (8, 12), 'failure': (15, 25)}
            },
            'compressor': {
                'temperature': {'normal': (80, 100), 'failure': (130, 160)},
                'vibration': {'normal': (0.2, 0.4), 'failure': (1.0, 2.0)},
                'pressure': {'normal': (8, 12), 'failure': (4, 7)},
                'flow_rate': {'normal': (200, 300), 'failure': (100, 180)},
                'power_consumption': {'normal': (15, 20), 'failure': (25, 35)}
            },
            'turbine': {
                'temperature': {'normal': (400, 500), 'failure': (600, 800)},
                'vibration': {'normal': (0.1, 0.25), 'failure': (0.7, 1.3)},
                'rpm': {'normal': (3000, 3600), 'failure': (2500, 2900)},
                'power_output': {'normal': (1000, 1200), 'failure': (600, 900)},
                'efficiency': {'normal': (85, 95), 'failure': (60, 80)}
            },
            'generator': {
                'temperature': {'normal': (60, 85), 'failure': (100, 130)},
                'vibration': {'normal': (0.05, 0.15), 'failure': (0.5, 1.0)},
                'voltage': {'normal': (380, 420), 'failure': (350, 370)},
                'current': {'normal': (100, 150), 'failure': (200, 300)},
                'power_output': {'normal': (500, 800), 'failure': (200, 400)}
            }
        }
    
    def generate_failure_pattern(self, duration_hours: int, failure_probability: float = 0.1) -> List[bool]:
        """Generate failure pattern over time"""
        # Create failure events with clustering (failures often come in groups)
        failure_pattern = []
        in_failure_period = False
        failure_duration = 0
        
        for hour in range(duration_hours):
            if not in_failure_period:
                # Check if failure starts
                if random.random() < failure_probability:
                    in_failure_period = True
                    failure_duration = random.randint(2, 24)  # Failure lasts 2-24 hours
                    failure_pattern.append(True)
                else:
                    failure_pattern.append(False)
            else:
                # Continue failure period
                failure_pattern.append(True)
                failure_duration -= 1
                if failure_duration <= 0:
                    in_failure_period = False
        
        return failure_pattern
    
    def add_noise_and_trends(self, base_value: float, time_step: int, 
                           trend_factor: float = 0.001, noise_factor: float = 0.05) -> float:
        """Add realistic noise and trends to sensor data"""
        # Add gradual trend (equipment degradation)
        trend = trend_factor * time_step
        
        # Add random noise
        noise = np.random.normal(0, noise_factor * base_value)
        
        # Add cyclic patterns (daily/weekly cycles)
        daily_cycle = 0.1 * np.sin(2 * np.pi * time_step / 24)
        weekly_cycle = 0.05 * np.sin(2 * np.pi * time_step / (24 * 7))
        
        return base_value + trend + noise + daily_cycle + weekly_cycle
    
    def generate_sensor_data(self, equipment_id: str, equipment_type: str, 
                           start_time: datetime, duration_hours: int,
                           sampling_rate_minutes: int = 15) -> pd.DataFrame:
        """Generate sensor data for a specific equipment"""
        
        if equipment_type not in self.sensor_configs:
            raise ValueError(f"Unknown equipment type: {equipment_type}")
        
        config = self.sensor_configs[equipment_type]
        failure_pattern = self.generate_failure_pattern(duration_hours)
        
        # Calculate number of data points
        points_per_hour = 60 // sampling_rate_minutes
        total_points = duration_hours * points_per_hour
        
        # Generate timestamps
        timestamps = []
        current_time = start_time
        for _ in range(total_points):
            timestamps.append(current_time)
            current_time += timedelta(minutes=sampling_rate_minutes)
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            hour_index = i // points_per_hour
            is_failing = failure_pattern[hour_index] if hour_index < len(failure_pattern) else False
            
            row = {
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'is_failure': is_failing
            }
            
            # Generate sensor readings
            for sensor_name, sensor_config in config.items():
                if is_failing:
                    base_range = sensor_config['failure']
                else:
                    base_range = sensor_config['normal']
                
                base_value = np.random.uniform(base_range[0], base_range[1])
                
                # Add realistic variations
                sensor_value = self.add_noise_and_trends(
                    base_value, i, 
                    trend_factor=0.0001 if not is_failing else 0.001,
                    noise_factor=0.02 if not is_failing else 0.1
                )
                
                row[sensor_name] = max(0, sensor_value)  # Ensure non-negative values
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_fleet_data(self, num_equipment: int = 10, 
                          duration_days: int = 30,
                          start_date: datetime = None) -> pd.DataFrame:
        """Generate data for a fleet of equipment"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=duration_days)
        
        all_data = []
        
        for i in range(num_equipment):
            equipment_type = random.choice(self.equipment_types)
            equipment_id = f"{equipment_type}_{i+1:03d}"
            
            # Vary failure probability by equipment type and age
            base_failure_prob = {
                'pump': 0.08, 'motor': 0.06, 'compressor': 0.12,
                'turbine': 0.15, 'generator': 0.10
            }
            
            failure_prob = base_failure_prob.get(equipment_type, 0.1)
            
            equipment_data = self.generate_sensor_data(
                equipment_id=equipment_id,
                equipment_type=equipment_type,
                start_time=start_date,
                duration_hours=duration_days * 24,
                sampling_rate_minutes=15
            )
            
            all_data.append(equipment_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def save_data(self, data: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save generated data to file"""
        if format.lower() == 'csv':
            data.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            data.to_json(filepath, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_failure_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate failure statistics from generated data"""
        stats = {}
        
        # Overall statistics
        total_records = len(data)
        failure_records = len(data[data['is_failure'] == True])
        stats['overall_failure_rate'] = failure_records / total_records
        
        # Per equipment type
        stats['by_equipment_type'] = {}
        for eq_type in data['equipment_type'].unique():
            type_data = data[data['equipment_type'] == eq_type]
            type_failures = len(type_data[type_data['is_failure'] == True])
            stats['by_equipment_type'][eq_type] = {
                'total_records': len(type_data),
                'failure_records': type_failures,
                'failure_rate': type_failures / len(type_data)
            }
        
        # Per equipment
        stats['by_equipment'] = {}
        for eq_id in data['equipment_id'].unique():
            eq_data = data[data['equipment_id'] == eq_id]
            eq_failures = len(eq_data[eq_data['is_failure'] == True])
            stats['by_equipment'][eq_id] = {
                'total_records': len(eq_data),
                'failure_records': eq_failures,
                'failure_rate': eq_failures / len(eq_data)
            }
        
        return stats


if __name__ == "__main__":
    # Example usage
    simulator = IoTSensorSimulator()
    
    # Generate data for a fleet
    print("Generating IoT sensor data...")
    fleet_data = simulator.generate_fleet_data(
        num_equipment=5,
        duration_days=7,
        start_date=datetime(2024, 1, 1)
    )
    
    print(f"Generated {len(fleet_data)} data points")
    print(f"Equipment types: {fleet_data['equipment_type'].unique()}")
    print(f"Date range: {fleet_data['timestamp'].min()} to {fleet_data['timestamp'].max()}")
    
    # Save data
    simulator.save_data(fleet_data, 'data/simulated_sensor_data.csv')
    
    # Print statistics
    stats = simulator.get_failure_statistics(fleet_data)
    print(f"\nFailure Statistics:")
    print(f"Overall failure rate: {stats['overall_failure_rate']:.2%}")
    
    for eq_type, type_stats in stats['by_equipment_type'].items():
        print(f"{eq_type}: {type_stats['failure_rate']:.2%}")
