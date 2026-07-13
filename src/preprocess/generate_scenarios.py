# =========================================================================
# generate_scenarios.py
# Multi-Step Supervised Sequence Dataset Generator 
# =========================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from preprocess.config import (
    FULL_GRID_TENSOR_FILE,
    TIMESTAMP_FILE,
    SCENARIOS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VALID_START_DATE,
    VALID_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    SCENARIO_OUTPUT_DIR,
    USE_MINMAX_SCALING
)

# =========================================================================
# LOAD MASTER DATASET
# =========================================================================
def load_master_dataset():
    print("\n[+] Loading master SLA geospatial tensor...")
    sla_tensor = np.load(FULL_GRID_TENSOR_FILE)
    timestamps = pd.to_datetime(np.load(TIMESTAMP_FILE))
    print(f"[+] Master Tensor Dimensions Locked: {sla_tensor.shape} (Days, H, W)")
    return sla_tensor, timestamps

# =========================================================================
# ROBUST SEQUENTIAL SLIDING WINDOW GENERATOR
# =========================================================================
def create_supervised_sequences(tensor, input_window, output_window):
    """
    Constructs multi-step input-target pairs using a sliding window.
    Strictly restricted within a single pre-cut chronological subset.
    """
    X, y = [], []
    total_days = len(tensor)
    
    max_start_idx = total_days - input_window - output_window + 1
    
    if max_start_idx <= 0:
        return np.array([]), np.array([])

    for start_idx in range(max_start_idx):
        end_input_idx = start_idx + input_window
        end_output_idx = end_input_idx + output_window
        
        x_seq = tensor[start_idx:end_input_idx]
        y_seq = tensor[end_input_idx:end_output_idx]
        
        X.append(x_seq)
        y.append(y_seq)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    if X.ndim == 4:
        X = np.expand_dims(X, axis=-1)
    if y.ndim == 4:
        y = np.expand_dims(y, axis=-1)
        
    return X, y

# =========================================================================
# MAIN COHESIVE PIPELINE GENERATOR
# =========================================================================
def generate_all_scenarios():
    print("=" * 70)
    print("INITIALIZING LEAKAGE-FREE SCENARIO GENERATION ENGINE")
    print("=" * 70)

    # LOAD MASTER DATA
    tensor, timestamps = load_master_dataset()
    df_time = pd.DataFrame({"timestamp": timestamps})

    # =====================================================================
    # CRITICAL RECONSTRUCTION: CHRONOLOGICAL PARTITIONING BEFORE WINDOWING
    # =====================================================================
    print("\n[+] Executing chronological split on raw master layers...")
    train_mask = (df_time["timestamp"] >= TRAIN_START_DATE) & (df_time["timestamp"] <= TRAIN_END_DATE)
    val_mask   = (df_time["timestamp"] >= VALID_START_DATE) & (df_time["timestamp"] <= VALID_END_DATE)
    test_mask  = (df_time["timestamp"] >= TEST_START_DATE)  & (df_time["timestamp"] <= TEST_END_DATE)

    raw_train_tensor = tensor[train_mask]
    raw_val_tensor   = tensor[val_mask]
    raw_test_tensor  = tensor[test_mask]

    print(f"    -> Isolated Train Block : {raw_train_tensor.shape} Days")
    print(f"    -> Isolated Val Block   : {raw_val_tensor.shape} Days")
    print(f"    -> Isolated Test Block  : {raw_test_tensor.shape} Days")

    # =====================================================================
    # DATA NORMALIZATION (GLOBAL TRAINING FIT ONLY)
    # Fits scaler strictly on training nodes to protect validation/test curves
    # =====================================================================
    if USE_MINMAX_SCALING:
        print("\n[+] Initializing global training-bound MinMaxScaler...")
        scaler = MinMaxScaler(feature_range=(-1, 1)) # Bounded domain for Tanh gate optimization
        
        train_finite_values = raw_train_tensor[np.isfinite(raw_train_tensor)].reshape(-1, 1)
        scaler.fit(train_finite_values)
        
        def scale_subset_tensor(target_tensor):
            shape_saved = target_tensor.shape
            flat_tensor = target_tensor.reshape(-1)
            nan_mask = np.isnan(flat_tensor)
            
            valid_extracted = flat_tensor[~nan_mask].reshape(-1, 1)
            scaled_extracted = scaler.transform(valid_extracted).flatten()
            
            output_flat = flat_tensor.copy()
            output_flat[~nan_mask] = scaled_extracted
            return output_flat.reshape(shape_saved).astype(np.float32)

        print("[+] Transforming isolated subsets into scaled [-1, 1] arrays...")
        train_tensor_scaled = scale_subset_tensor(raw_train_tensor)
        val_tensor_scaled   = scale_subset_tensor(raw_val_tensor)
        test_tensor_scaled  = scale_subset_tensor(raw_test_tensor)
    else:
        print("\n[!] Scaling disabled via config. Copying raw sub-tensors.")
        train_tensor_scaled = raw_train_tensor.copy()
        val_tensor_scaled   = raw_val_tensor.copy()
        test_tensor_scaled  = raw_test_tensor.copy()
        scaler = None

    # =====================================================================
    # ITERATION LOOP OVER EXPERIMENT MATRIX SCENARIOS
    # =====================================================================
    for input_window, output_window in tqdm(SCENARIOS, desc="Compiling Scenario Pipelines"):
        scenario_name = f"scenario_{input_window}_{output_window}"
        scenario_dir = os.path.join(SCENARIO_OUTPUT_DIR, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        X_train, y_train = create_supervised_sequences(train_tensor_scaled, input_window, output_window)
        X_val, y_val     = create_supervised_sequences(val_tensor_scaled, input_window, output_window)
        X_test, y_test   = create_supervised_sequences(test_tensor_scaled, input_window, output_window)

        npz_save_path = os.path.join(scenario_dir, f"{scenario_name}_compressed.npz")
        np.savez_compressed(
            npz_save_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )

        if scaler is not None:
            joblib.dump(scaler, os.path.join(scenario_dir, "scaler.pkl"))

        metadata = {
            "scenario_name": scenario_name,
            "input_window_T": input_window,
            "output_horizon_H": output_window,
            "samples_count": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test))
            },
            "tensor_shapes": {
                "X_train": list(X_train.shape),
                "y_train": list(y_train.shape)
            },
            "scaling_applied": USE_MINMAX_SCALING
        }

        with open(os.path.join(scenario_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUCCESSFUL: ALL SCENARIOS SECURED FROM LEAKAGE")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_scenarios()