#!/usr/bin/env python3
"""
Data Pipeline Runner
Runs the complete data ingestion and processing pipeline
"""

import os
import sys
import logging
import time
import signal
from typing import List

# Add src to path
sys.path.append('/app')

from src.pipeline import DataPipeline, SensorReading
from src.services import PredictionServiceManager
from src.data_simulation import IoTSensorSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline runner"""
    
    def __init__(self):
        self.pipeline = None
        self.prediction_service = None
        self.running = False
        
    def setup_services(self):
        """Initialize services"""
        try:
            # Get configuration from environment
            kafka_servers = os.getenv('KAFKA_SERVERS', 'localhost:9092').split(',')
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            
            # Initialize pipeline
            self.pipeline = DataPipeline(
                kafka_servers=kafka_servers,
                redis_host=redis_host,
                redis_port=redis_port
            )
            
            # Initialize prediction service
            self.prediction_service = PredictionServiceManager(
                models_dir='/app/models',
                redis_cache=self.pipeline.redis_cache
            )
            
            # Try to load existing models
            try:
                self.prediction_service.initialize()
                logger.info("Loaded existing models")
            except Exception as e:
                logger.warning(f"Could not load models: {e}")
                logger.info("Pipeline will run without predictions until models are trained")
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup services: {e}")
            raise
    
    def prediction_callback(self, equipment_id: str, recent_data: List[SensorReading]):
        """Callback for making predictions"""
        try:
            if self.prediction_service and self.prediction_service.prediction_service.lstm_model.is_trained:
                result = self.prediction_service.predict(equipment_id, recent_data)
                
                # Log prediction result
                if 'predictions' in result and 'ensemble' in result['predictions']:
                    prob = result['predictions']['ensemble']['failure_probability']
                    confidence = result['predictions']['ensemble'].get('confidence', 0.0)
                    
                    logger.info(f"Prediction for {equipment_id}: "
                              f"failure_prob={prob:.3f}, confidence={confidence:.3f}")
                    
                    # Alert on high failure probability
                    if prob > 0.8:
                        logger.warning(f"HIGH FAILURE RISK for {equipment_id}: {prob:.3f}")
                else:
                    logger.debug(f"Prediction completed for {equipment_id}")
            else:
                logger.debug(f"Skipping prediction for {equipment_id} - models not ready")
                
        except Exception as e:
            logger.error(f"Prediction callback error for {equipment_id}: {e}")
    
    def start_pipeline(self):
        """Start the data pipeline"""
        try:
            # Add prediction callback
            self.pipeline.add_prediction_callback(self.prediction_callback)
            
            # Start pipeline
            consumer_thread = self.pipeline.start_pipeline()
            
            # Start simulation if enabled
            simulate_data = os.getenv('SIMULATE_DATA', 'true').lower() == 'true'
            num_equipment = int(os.getenv('NUM_EQUIPMENT', '10'))
            interval_seconds = int(os.getenv('SIMULATION_INTERVAL', '15'))
            
            simulation_thread = None
            if simulate_data:
                logger.info(f"Starting data simulation with {num_equipment} equipment")
                simulation_thread = self.pipeline.start_simulation(
                    num_equipment=num_equipment,
                    interval_seconds=interval_seconds
                )
            
            self.running = True
            logger.info("Pipeline started successfully")
            
            # Keep running
            try:
                while self.running:
                    time.sleep(10)
                    
                    # Check thread health
                    if consumer_thread and not consumer_thread.is_alive():
                        logger.error("Consumer thread died, restarting...")
                        consumer_thread = self.pipeline.start_pipeline()
                    
                    if simulation_thread and simulate_data and not simulation_thread.is_alive():
                        logger.error("Simulation thread died, restarting...")
                        simulation_thread = self.pipeline.start_simulation(
                            num_equipment=num_equipment,
                            interval_seconds=interval_seconds
                        )
                        
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.stop_pipeline()
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        self.running = False
        if self.pipeline:
            self.pipeline.stop_pipeline()
        logger.info("Pipeline stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_pipeline()
        sys.exit(0)


def main():
    """Main function"""
    runner = PipelineRunner()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, runner.signal_handler)
    signal.signal(signal.SIGTERM, runner.signal_handler)
    
    try:
        # Wait for dependencies to be ready
        logger.info("Waiting for dependencies...")
        time.sleep(30)  # Give time for Kafka, Redis, etc. to start
        
        # Setup and start
        runner.setup_services()
        runner.start_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
