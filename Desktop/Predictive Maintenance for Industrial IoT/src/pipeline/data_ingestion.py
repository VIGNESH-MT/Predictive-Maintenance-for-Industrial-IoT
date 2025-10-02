"""
Real-time Data Ingestion Pipeline
Handles streaming IoT sensor data using Kafka and Redis
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Data class for sensor readings"""
    timestamp: datetime
    equipment_id: str
    equipment_type: str
    sensor_data: Dict[str, float]
    is_failure: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SensorReading':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class DataProducer:
    """Kafka producer for streaming sensor data"""
    
    def __init__(self, kafka_servers: List[str] = ['localhost:9092'], 
                 topic: str = 'sensor_data'):
        self.kafka_servers = kafka_servers
        self.topic = topic
        self.producer = None
        self.is_running = False
        
    def connect(self):
        """Connect to Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Connected to Kafka producer")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_sensor_data(self, sensor_reading: SensorReading):
        """Send sensor reading to Kafka topic"""
        if not self.producer:
            raise ValueError("Producer not connected")
        
        try:
            # Use equipment_id as key for partitioning
            key = sensor_reading.equipment_id
            value = sensor_reading.to_dict()
            
            future = self.producer.send(self.topic, key=key, value=value)
            future.get(timeout=10)  # Wait for confirmation
            
        except Exception as e:
            logger.error(f"Failed to send sensor data: {e}")
            raise
    
    def start_simulation_stream(self, simulator, num_equipment: int = 10, 
                              interval_seconds: int = 15):
        """Start streaming simulated sensor data"""
        from src.data_simulation import IoTSensorSimulator
        
        if not self.producer:
            self.connect()
        
        self.is_running = True
        logger.info(f"Starting simulation stream for {num_equipment} equipment")
        
        # Generate initial equipment list
        equipment_list = []
        for i in range(num_equipment):
            equipment_type = np.random.choice(['pump', 'motor', 'compressor', 'turbine', 'generator'])
            equipment_list.append({
                'id': f"{equipment_type}_{i+1:03d}",
                'type': equipment_type
            })
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                for equipment in equipment_list:
                    # Generate single data point for this equipment
                    data = simulator.generate_sensor_data(
                        equipment_id=equipment['id'],
                        equipment_type=equipment['type'],
                        start_time=current_time,
                        duration_hours=1,
                        sampling_rate_minutes=60  # Single point
                    )
                    
                    if not data.empty:
                        row = data.iloc[0]
                        
                        # Extract sensor data
                        sensor_columns = [col for col in data.columns 
                                        if col not in ['timestamp', 'equipment_id', 'equipment_type', 'is_failure']]
                        sensor_data = {col: float(row[col]) for col in sensor_columns}
                        
                        # Create sensor reading
                        reading = SensorReading(
                            timestamp=current_time,
                            equipment_id=equipment['id'],
                            equipment_type=equipment['type'],
                            sensor_data=sensor_data,
                            is_failure=bool(row['is_failure'])
                        )
                        
                        # Send to Kafka
                        self.send_sensor_data(reading)
                
                logger.info(f"Sent data for {len(equipment_list)} equipment units")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Simulation stream stopped by user")
        except Exception as e:
            logger.error(f"Error in simulation stream: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the producer"""
        self.is_running = False
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


class DataConsumer:
    """Kafka consumer for processing sensor data"""
    
    def __init__(self, kafka_servers: List[str] = ['localhost:9092'],
                 topic: str = 'sensor_data',
                 group_id: str = 'sensor_processor'):
        self.kafka_servers = kafka_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.is_running = False
        self.message_handlers: List[Callable] = []
        
    def connect(self):
        """Connect to Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest'
            )
            logger.info(f"Connected to Kafka consumer for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka consumer: {e}")
            raise
    
    def add_message_handler(self, handler: Callable[[SensorReading], None]):
        """Add a message handler function"""
        self.message_handlers.append(handler)
    
    def start_consuming(self):
        """Start consuming messages"""
        if not self.consumer:
            self.connect()
        
        self.is_running = True
        logger.info("Starting message consumption")
        
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                try:
                    # Parse sensor reading
                    sensor_reading = SensorReading.from_dict(message.value)
                    
                    # Process with all handlers
                    for handler in self.message_handlers:
                        try:
                            handler(sensor_reading)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the consumer"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


class RedisCache:
    """Redis cache for storing recent sensor data and predictions"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: str = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis_client = None
        
    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def store_sensor_data(self, sensor_reading: SensorReading, ttl_seconds: int = 86400):
        """Store sensor reading in Redis with TTL"""
        if not self.redis_client:
            self.connect()
        
        try:
            key = f"sensor:{sensor_reading.equipment_id}:{sensor_reading.timestamp.isoformat()}"
            value = json.dumps(sensor_reading.to_dict())
            
            self.redis_client.setex(key, ttl_seconds, value)
            
            # Also maintain a sorted set for time-based queries
            equipment_key = f"equipment:{sensor_reading.equipment_id}"
            timestamp_score = sensor_reading.timestamp.timestamp()
            self.redis_client.zadd(equipment_key, {key: timestamp_score})
            
            # Set TTL on the sorted set as well
            self.redis_client.expire(equipment_key, ttl_seconds)
            
        except Exception as e:
            logger.error(f"Failed to store sensor data in Redis: {e}")
    
    def get_recent_data(self, equipment_id: str, hours: int = 24) -> List[SensorReading]:
        """Get recent sensor data for equipment"""
        if not self.redis_client:
            self.connect()
        
        try:
            equipment_key = f"equipment:{equipment_id}"
            
            # Get timestamps from last N hours
            end_time = datetime.now().timestamp()
            start_time = (datetime.now() - timedelta(hours=hours)).timestamp()
            
            # Get keys from sorted set
            keys = self.redis_client.zrangebyscore(equipment_key, start_time, end_time)
            
            # Retrieve sensor readings
            readings = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    reading_dict = json.loads(data)
                    readings.append(SensorReading.from_dict(reading_dict))
            
            return sorted(readings, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to get recent data from Redis: {e}")
            return []
    
    def store_prediction(self, equipment_id: str, prediction_data: Dict, ttl_seconds: int = 3600):
        """Store prediction results"""
        if not self.redis_client:
            self.connect()
        
        try:
            key = f"prediction:{equipment_id}"
            value = json.dumps(prediction_data, default=str)
            self.redis_client.setex(key, ttl_seconds, value)
            
        except Exception as e:
            logger.error(f"Failed to store prediction in Redis: {e}")
    
    def get_prediction(self, equipment_id: str) -> Optional[Dict]:
        """Get latest prediction for equipment"""
        if not self.redis_client:
            self.connect()
        
        try:
            key = f"prediction:{equipment_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get prediction from Redis: {e}")
            return None


class DataProcessor:
    """Process incoming sensor data and trigger predictions"""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis_cache = redis_cache
        self.buffer = {}  # Buffer for batching data
        self.buffer_size = 100
        self.prediction_callbacks: List[Callable] = []
        
    def add_prediction_callback(self, callback: Callable[[str, List[SensorReading]], None]):
        """Add callback for triggering predictions"""
        self.prediction_callbacks.append(callback)
    
    def process_sensor_reading(self, sensor_reading: SensorReading):
        """Process a single sensor reading"""
        try:
            # Store in Redis
            self.redis_cache.store_sensor_data(sensor_reading)
            
            # Add to buffer
            equipment_id = sensor_reading.equipment_id
            if equipment_id not in self.buffer:
                self.buffer[equipment_id] = []
            
            self.buffer[equipment_id].append(sensor_reading)
            
            # Check if we should trigger prediction
            if len(self.buffer[equipment_id]) >= 10:  # Trigger every 10 readings
                recent_data = self.buffer[equipment_id][-24:]  # Last 24 readings
                
                # Trigger prediction callbacks
                for callback in self.prediction_callbacks:
                    try:
                        callback(equipment_id, recent_data)
                    except Exception as e:
                        logger.error(f"Prediction callback error: {e}")
                
                # Clear buffer for this equipment
                self.buffer[equipment_id] = []
            
            logger.debug(f"Processed sensor reading for {equipment_id}")
            
        except Exception as e:
            logger.error(f"Error processing sensor reading: {e}")


class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, kafka_servers: List[str] = ['localhost:9092'],
                 redis_host: str = 'localhost', redis_port: int = 6379):
        self.kafka_servers = kafka_servers
        self.redis_cache = RedisCache(host=redis_host, port=redis_port)
        self.data_processor = DataProcessor(self.redis_cache)
        self.consumer = DataConsumer(kafka_servers=kafka_servers)
        self.producer = DataProducer(kafka_servers=kafka_servers)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def add_prediction_callback(self, callback: Callable):
        """Add prediction callback"""
        self.data_processor.add_prediction_callback(callback)
    
    def start_pipeline(self):
        """Start the complete data pipeline"""
        try:
            # Connect to services
            self.redis_cache.connect()
            
            # Add message handler
            self.consumer.add_message_handler(self.data_processor.process_sensor_reading)
            
            # Start consumer in separate thread
            consumer_thread = threading.Thread(target=self.consumer.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()
            
            logger.info("Data pipeline started successfully")
            return consumer_thread
            
        except Exception as e:
            logger.error(f"Failed to start data pipeline: {e}")
            raise
    
    def start_simulation(self, num_equipment: int = 10, interval_seconds: int = 15):
        """Start data simulation"""
        from src.data_simulation import IoTSensorSimulator
        
        simulator = IoTSensorSimulator()
        
        # Start simulation in separate thread
        simulation_thread = threading.Thread(
            target=self.producer.start_simulation_stream,
            args=(simulator, num_equipment, interval_seconds)
        )
        simulation_thread.daemon = True
        simulation_thread.start()
        
        return simulation_thread
    
    def stop_pipeline(self):
        """Stop the data pipeline"""
        self.consumer.stop()
        self.producer.stop()
        self.executor.shutdown(wait=True)
        logger.info("Data pipeline stopped")


if __name__ == "__main__":
    # Example usage
    def prediction_callback(equipment_id: str, recent_data: List[SensorReading]):
        """Example prediction callback"""
        logger.info(f"Triggering prediction for {equipment_id} with {len(recent_data)} readings")
    
    # Create and start pipeline
    pipeline = DataPipeline()
    pipeline.add_prediction_callback(prediction_callback)
    
    try:
        # Start pipeline
        consumer_thread = pipeline.start_pipeline()
        
        # Start simulation
        simulation_thread = pipeline.start_simulation(num_equipment=3, interval_seconds=10)
        
        # Keep running
        consumer_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Stopping pipeline...")
        pipeline.stop_pipeline()
