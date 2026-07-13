# =========================================================================
# main.py
# Master Orchestration Pipeline for Coastal SLA Forecasting Dataset Preparation
# Fully Synchronized with Leakage-Free and Compressed .NPZ Data Pipelines
# =========================================================================

import os
from pathlib import Path

from preprocess.config import (
    FULL_GRID_TENSOR_FILE,
    TIMESTAMP_FILE,
    COASTAL_POINT_FILE,
    METADATA_FILE,
    SCENARIO_OUTPUT_DIR,
    SCENARIOS
)
from preprocess.build_sla_dataset import build_sla_dataset
from preprocess.generate_scenarios import generate_all_scenarios


# =========================================================================
# SMART CACHING VERIFICATION HELPERS
# =========================================================================
def dataset_outputs_exist():
    """
    Verifies if core processed grid tensors and coordinate shapes exist on disk.
    """
    required_outputs = [
        FULL_GRID_TENSOR_FILE,
        TIMESTAMP_FILE,
        COASTAL_POINT_FILE,
        METADATA_FILE
    ]
    return all(Path(path).exists() for path in required_outputs)


def scenario_outputs_exist():
    """
    Strictly validates the presence of compressed .npz matrices
    across ALL scenarios to prevent unzipping/bypass failures during Kaggle upload.
    """
    scenario_base_dir = Path(SCENARIO_OUTPUT_DIR)
    if not scenario_base_dir.exists():
        return False

    valid_scenario_count = 0
    
    for input_window, output_window in SCENARIOS:
        scenario_name = f"scenario_{input_window}_{output_window}"
        target_npz = scenario_base_dir / scenario_name / f"{scenario_name}_compressed.npz"
        
        if target_npz.exists():
            valid_scenario_count += 1

    return valid_scenario_count == len(SCENARIOS)


# =========================================================================
# MAIN EXECUTIVE MASTER PIPELINE
# =========================================================================
def main():
    print("=" * 75)
    print("      COASTAL SPATIO-TEMPORAL SLA FORECASTING PREPROCESSING PIPELINE")
    print("=" * 75)

    # ---------------------------------------------------------------------
    # STEP 1 — SUBSAMPLED GRID TENSOR CONSTRUCTION 
    # ---------------------------------------------------------------------
    print("\n[STEP 1] PARSING SUBSET COORDINATES & BUILDING MASTER SLA TENSOR...")
    if dataset_outputs_exist():
        print("[+] Cache Hit: Full-grid master tensors and metadata maps already exist on disk.")
        print("[+] Skipping structural dataset rebuilding.")
    else:
        build_sla_dataset()

    # ---------------------------------------------------------------------
    # STEP 2 — ROBUST SCENARIO SEQUENCE ARTIFACT GENERATION
    # ---------------------------------------------------------------------
    print("\n[STEP 2] COMPILING MULTI-STEP FORECASTING SCENARIO MATRIX...")
    if scenario_outputs_exist():
        print("[+] Cache Hit: All experimental scenario compressed .npz files fully intact.")
        print("[+] Skipping sliding-window dataset generation.")
    else:
        print("[!] Missing or incomplete scenario layers identified. Starting compilation...")
        generate_all_scenarios()

    # ---------------------------------------------------------------------
    # FINAL PRODUCTION LOG SUMMARY
    # ---------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("ENTIRE MASTER DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  >>> Zip and upload the directory: '{SCENARIO_OUTPUT_DIR}' to Kaggle Data Datasets.")
    print("=" * 75)


if __name__ == "__main__":
    main()