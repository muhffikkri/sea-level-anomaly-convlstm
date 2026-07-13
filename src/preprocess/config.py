# =========================================================================
# config.py
# Centralized Configuration for Spatio-Temporal Coastal SLA Forecasting
# =========================================================================

import os
from pathlib import Path

# =========================================================================
# PROJECT DIRECTORY ROOT
# =========================================================================
PROJECT_ROOT = r"sea-level-anomaly-convlstm"

# =========================================================================
# RAW GEOSPATIAL DATA CONFIGURATION
# =========================================================================
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
NETCDF_EXTENSION = ".nc"
VARIABLE_NAME = "sla"

# =========================================================================
# STUDY AREA CONFIGURATION
# Covering macro-regional Central Java shoreline (109.5°E - 111.5°E)
# =========================================================================
MIN_LON = 109.5
MAX_LON = 111.5

MIN_LAT = -8.0  
MAX_LAT = -6.0  

GRID_RESOLUTION = 0.125
GRID_H = 16
GRID_W = 16

# =========================================================================
# OUTPUT DIRECTORIES GENERATION
# =========================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

PROCESSED_OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
REFERENCE_OUTPUT_DIR = os.path.join(DATA_DIR, "reference")
SCENARIO_OUTPUT_DIR  = os.path.join(DATA_DIR, "scenarios")
METADATA_OUTPUT_DIR  = os.path.join(DATA_DIR, "metadata")
FIGURE_OUTPUT_DIR    = os.path.join(DATA_DIR, "figures")
MODEL_OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models")
PRED_OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "predictions")

# =========================================================================
# CORE TARGET GENERATED ARTIFACTS
# =========================================================================
FULL_GRID_TENSOR_FILE = os.path.join(PROCESSED_OUTPUT_DIR, "sla_tensor.npy")
TIMESTAMP_FILE        = os.path.join(PROCESSED_OUTPUT_DIR, "timestamps.npy")
COASTAL_POINT_FILE    = os.path.join(REFERENCE_OUTPUT_DIR, "coastal_points.parquet")
METADATA_FILE         = os.path.join(METADATA_OUTPUT_DIR, "metadata.json")

# =========================================================================
# CHRONOLOGICAL TEMPORAL SPLIT 
# Train: 2016-2021 | Validation: 2022-2023 | Testing: 2024-2025
# =========================================================================
TRAIN_START_DATE = "2016-01-01"
TRAIN_END_DATE   = "2021-12-31"

VALID_START_DATE = "2022-01-01"
VALID_END_DATE   = "2023-12-31"

TEST_START_DATE  = "2024-01-01"
TEST_END_DATE    = "2025-12-31"

# =========================================================================
# SCALER CONFIGURATION
# Bounded domain for deep learning tanh gate operational stability
# =========================================================================
USE_MINMAX_SCALING = True  # Scaler must be fitted on Training Set ONLY

# =========================================================================
# FORECASTING SCENARIO MATRIX (COMPREHENSIVE EXPANDED LIST)
# Fully charts input temporal windows (T) vs output prediction horizons (H)
# =========================================================================
SCENARIOS = [
    # Long Historical Context (T=60)
    (60, 30), (60, 15), (60, 7), (60, 3), (60, 1),
    
    # Medium Historical Context (T=30)
    (30, 30), (30, 15), (30, 7), (30, 3), (30, 1),
    
    # Short Historical Context (T=15)
    (15, 7), (15, 3), (15, 1),
    
    # Minimal Historical Context (T=7)
    (7, 3), (7, 1),
    
    # Near Persistence-like Context (T=3)
    (3, 1)
]

# =========================================================================
# EXTREME COASTAL EVENT HAZARD CONFIGURATION
# Hazard calculations bound dynamically to the P95 training distribution
# =========================================================================
EXTREME_PERCENTILE = 95
EXTREME_WEIGHT = 3.0

# =========================================================================
# REPRODUCIBILITY & TRAINING HYPERPARAMETERS
# =========================================================================
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
ENABLE_GPU_MEMORY_GROWTH = True

# =========================================================================
# EVALUATION SEQUENCES & ARCHITECTURAL BASELINES
# Fully integrated to map spatial benefits and persistence scores
# =========================================================================
EVALUATION_METRICS = ["RMSE", "MAE", "MSE", "Extreme_RMSE", "Extreme_MAE", "PSS"]
BASELINE_MODELS   = ["Persistence", "XGBoost", "LSTM"]
PROPOSED_MODELS   = ["ConvLSTM", "Weighted_ConvLSTM"]

# =========================================================================
# AUTOMATIC INSTANTIATION OF ENVIRONMENT DIRECTORIES
# =========================================================================
ALL_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_OUTPUT_DIR,
    REFERENCE_OUTPUT_DIR,
    SCENARIO_OUTPUT_DIR,
    METADATA_OUTPUT_DIR,
    FIGURE_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
    PRED_OUTPUT_DIR
]

for directory in ALL_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

print("=" * 70)
print("config.py: Centralized Configuration Locked & Aligned with Reviewer Blueprint")
print("=" * 70)