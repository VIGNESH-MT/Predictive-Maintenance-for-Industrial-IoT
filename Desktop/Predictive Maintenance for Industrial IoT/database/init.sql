-- Initialize database schema for Predictive Maintenance System

-- Create sensor_data table
CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(50) NOT NULL,
    temperature FLOAT,
    vibration FLOAT,
    pressure FLOAT,
    flow_rate FLOAT,
    power_consumption FLOAT,
    current FLOAT,
    rpm FLOAT,
    power_output FLOAT,
    efficiency FLOAT,
    voltage FLOAT,
    is_failure BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(50),
    failure_probability FLOAT NOT NULL,
    predicted_failure BOOLEAN NOT NULL,
    confidence FLOAT,
    model_used VARCHAR(50),
    prediction_horizon_hours INTEGER,
    lstm_probability FLOAT,
    arima_probability FLOAT,
    ensemble_probability FLOAT,
    data_points INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    mse FLOAT,
    mae FLOAT,
    equipment_type VARCHAR(50),
    training_samples INTEGER,
    validation_samples INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create maintenance_events table
CREATE TABLE IF NOT EXISTS maintenance_events (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'scheduled', 'predictive', 'reactive'
    event_date TIMESTAMP WITH TIME ZONE NOT NULL,
    description TEXT,
    cost DECIMAL(10,2),
    downtime_hours FLOAT,
    was_predicted BOOLEAN DEFAULT FALSE,
    prediction_accuracy FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create equipment_metadata table
CREATE TABLE IF NOT EXISTS equipment_metadata (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_type VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    installation_date DATE,
    location VARCHAR(100),
    criticality_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    maintenance_schedule VARCHAR(50),
    last_maintenance DATE,
    next_scheduled_maintenance DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_sensor_data_equipment_id ON sensor_data(equipment_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_equipment_type ON sensor_data(equipment_type);
CREATE INDEX IF NOT EXISTS idx_sensor_data_is_failure ON sensor_data(is_failure);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_equipment_id ON predictions(equipment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_failure_probability ON predictions(failure_probability);

CREATE INDEX IF NOT EXISTS idx_maintenance_events_equipment_id ON maintenance_events(equipment_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_event_date ON maintenance_events(event_date);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_event_type ON maintenance_events(event_type);

-- Create views for common queries
CREATE OR REPLACE VIEW equipment_health_summary AS
SELECT 
    s.equipment_id,
    s.equipment_type,
    COUNT(*) as total_readings,
    AVG(s.temperature) as avg_temperature,
    AVG(s.vibration) as avg_vibration,
    AVG(s.pressure) as avg_pressure,
    AVG(s.power_consumption) as avg_power_consumption,
    COUNT(CASE WHEN s.is_failure THEN 1 END) as failure_count,
    MAX(s.timestamp) as last_reading,
    COALESCE(p.failure_probability, 0) as latest_failure_probability,
    COALESCE(p.predicted_failure, false) as predicted_failure
FROM sensor_data s
LEFT JOIN LATERAL (
    SELECT failure_probability, predicted_failure
    FROM predictions 
    WHERE equipment_id = s.equipment_id 
    ORDER BY timestamp DESC 
    LIMIT 1
) p ON true
WHERE s.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY s.equipment_id, s.equipment_type, p.failure_probability, p.predicted_failure;

CREATE OR REPLACE VIEW failure_prediction_accuracy AS
SELECT 
    p.equipment_id,
    p.equipment_type,
    p.timestamp as prediction_time,
    p.failure_probability,
    p.predicted_failure,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM sensor_data s 
            WHERE s.equipment_id = p.equipment_id 
            AND s.is_failure = true 
            AND s.timestamp BETWEEN p.timestamp AND p.timestamp + INTERVAL '6 hours'
        ) THEN true 
        ELSE false 
    END as actual_failure,
    CASE 
        WHEN p.predicted_failure = EXISTS (
            SELECT 1 FROM sensor_data s 
            WHERE s.equipment_id = p.equipment_id 
            AND s.is_failure = true 
            AND s.timestamp BETWEEN p.timestamp AND p.timestamp + INTERVAL '6 hours'
        ) THEN true 
        ELSE false 
    END as prediction_correct
FROM predictions p
WHERE p.timestamp >= NOW() - INTERVAL '7 days';

-- Insert sample equipment metadata
INSERT INTO equipment_metadata (equipment_id, equipment_type, manufacturer, model, installation_date, location, criticality_level, maintenance_schedule) VALUES
('pump_001', 'pump', 'Grundfos', 'CR-150', '2020-01-15', 'Building A - Floor 1', 'high', 'monthly'),
('motor_001', 'motor', 'Siemens', '1LA7-163', '2019-06-20', 'Building A - Floor 2', 'medium', 'quarterly'),
('compressor_001', 'compressor', 'Atlas Copco', 'GA-75', '2021-03-10', 'Building B - Basement', 'critical', 'weekly'),
('turbine_001', 'turbine', 'GE', 'LM2500', '2018-11-05', 'Power Plant - Unit 1', 'critical', 'monthly'),
('generator_001', 'generator', 'Caterpillar', 'C32', '2020-08-12', 'Power Plant - Unit 2', 'high', 'monthly')
ON CONFLICT (equipment_id) DO NOTHING;

-- Create function to update equipment metadata timestamp
CREATE OR REPLACE FUNCTION update_equipment_metadata_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for equipment metadata updates
DROP TRIGGER IF EXISTS trigger_update_equipment_metadata_timestamp ON equipment_metadata;
CREATE TRIGGER trigger_update_equipment_metadata_timestamp
    BEFORE UPDATE ON equipment_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_equipment_metadata_timestamp();

-- Grant permissions (adjust as needed for your security requirements)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO predictive_maintenance_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO predictive_maintenance_user;
