-- ==============================================================================
-- Krishnetra Aavishkar: Precision Pest Management Experiment Database Schema
-- File: src/research/experiment_schema.sql
-- Description: Standardized schema for bioassay trials, environmental logs,
--              and response metrics. No fake results pre-populated.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS acoustic_experiments (
    experiment_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    hypothesis TEXT NOT NULL,
    lead_researcher VARCHAR(128) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(32) DEFAULT 'PLANNED', -- PLANNED, IN_PROGRESS, COMPLETED, CANCELLED
    notes TEXT
);

CREATE TABLE IF NOT EXISTS bioassay_trials (
    trial_id VARCHAR(64) PRIMARY KEY,
    experiment_id VARCHAR(64) REFERENCES acoustic_experiments(experiment_id) ON DELETE CASCADE,
    pest_class VARCHAR(64) NOT NULL, -- army_worm, corn_borer, blister_beetle
    ip102_class_id INTEGER NOT NULL,
    insect_stage VARCHAR(32) NOT NULL, -- adult_moth, 3rd_instar_larvae, adult_beetle
    specimen_id VARCHAR(64) NOT NULL,
    treatment_type VARCHAR(64) NOT NULL, -- CONTROL_SILENT, CONTINUOUS_US, SWEPT_PULSE_US, LOW_FREQ_VIB
    frequency_hz NUMERIC(10, 2), -- e.g. 35000.00 or NULL if silent
    frequency_sweep_min_hz NUMERIC(10, 2),
    frequency_sweep_max_hz NUMERIC(10, 2),
    pulse_repetition_rate_hz NUMERIC(6, 2),
    amplitude_db_spl NUMERIC(6, 2), -- Measured Sound Pressure Level at specimen position
    distance_meters NUMERIC(6, 3) NOT NULL DEFAULT 0.500,
    exposure_duration_sec NUMERIC(8, 2) NOT NULL,
    temperature_c NUMERIC(4, 1) NOT NULL,
    humidity_percent NUMERIC(4, 1) NOT NULL,
    lux_level NUMERIC(6, 2) DEFAULT 0.0,
    baseline_activity_score NUMERIC(8, 4), -- Normalized pre-stimulus movement velocity (mm/s)
    treatment_activity_score NUMERIC(8, 4), -- Normalized post-stimulus movement velocity (mm/s)
    time_spent_treated_zone_sec NUMERIC(8, 2) DEFAULT 0.0,
    time_spent_control_zone_sec NUMERIC(8, 2) DEFAULT 0.0,
    entries_to_treated_zone INTEGER DEFAULT 0,
    exits_from_treated_zone INTEGER DEFAULT 0,
    avoidance_preference_index NUMERIC(6, 4), -- [-1.0, +1.0] API
    response_latency_sec NUMERIC(6, 3), -- Time to initial startle / trajectory deviation
    feeding_suppression_pct NUMERIC(5, 2), -- % reduction in leaf disc consumption
    mortality_observed BOOLEAN DEFAULT FALSE,
    result_classification VARCHAR(64), -- REPELLED, NEUTRAL, ATTRACTED, HABITUATED, INCONCLUSIVE
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trial_tracking_points (
    point_id BIGSERIAL PRIMARY KEY,
    trial_id VARCHAR(64) REFERENCES bioassay_trials(trial_id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms NUMERIC(10, 2) NOT NULL,
    centroid_x NUMERIC(8, 2) NOT NULL,
    centroid_y NUMERIC(8, 2) NOT NULL,
    current_zone VARCHAR(32) NOT NULL, -- ZONE_TREATED, ZONE_CONTROL, ZONE_NEUTRAL
    velocity_mm_per_sec NUMERIC(8, 2),
    is_stimulus_active BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_bioassay_pest ON bioassay_trials(pest_class);
CREATE INDEX IF NOT EXISTS idx_bioassay_treatment ON bioassay_trials(treatment_type);
CREATE INDEX IF NOT EXISTS idx_bioassay_experiment ON bioassay_trials(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tracking_trial ON trial_tracking_points(trial_id, frame_number);
