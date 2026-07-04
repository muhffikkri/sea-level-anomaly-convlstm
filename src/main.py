# =========================================================================
# main.py
# Master Orchestration Pipeline for Coastal SLA Forecasting Dataset Preparation
# Fully Synchronized with Leakage-Free and Compressed .NPZ Data Pipelines
# =========================================================================

import os
from pathlib import Path

from config import (
    FULL_GRID_TENSOR_FILE,
    TIMESTAMP_FILE,
    COASTAL_POINT_FILE,
    METADATA_FILE,
    SCENARIO_OUTPUT_DIR,
    SCENARIOS
)
from integrity_check import check_dataset_integrity
from build_sla_dataset import build_sla_dataset
from exploratory_data_analysis import perform_eda
from generate_scenarios import generate_all_scenarios


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
    CORRECTED: Strictly validates the presence of compressed .npz matrices
    across ALL scenarios to prevent unzipping/bypass failures during Kaggle upload.
    """
    scenario_base_dir = Path(SCENARIO_OUTPUT_DIR)
    if not scenario_base_dir.exists():
        return False

    # Hitung jumlah skenario valid yang wajib terisi berkas compressed .npz
    valid_scenario_count = 0
    
    for input_window, output_window in SCENARIOS:
        scenario_name = f"scenario_{input_window}_{output_window}"
        target_npz = scenario_base_dir / scenario_name / f"{scenario_name}_compressed.npz"
        
        if target_npz.exists():
            valid_scenario_count += 1

    # Pipeline dianggap aman jika seluruh skenario (16 skenario) sukses terarsip .npz
    return valid_scenario_count == len(SCENARIOS)


# =========================================================================
# MAIN EXECUTIVE MASTER PIPELINE
# =========================================================================
def main():
    print("=" * 75)
    print("      COASTAL SPATIO-TEMPORAL SLA FORECASTING PREPROCESSING PIPELINE")
    print("=" * 75)

    # ---------------------------------------------------------------------
    # STEP 1 — RAW GEOSPATIAL NETCDF INTEGRITY CHECK
    # ---------------------------------------------------------------------
    print("\n[STEP 1] EXECUTING RAW NETCDF DATA QUALITY & INTEGRITY CHECK...")
    all_files = check_dataset_integrity()
    
    if all_files is None or len(all_files) == 0:
        print("\n[-] Critical Failure: Dataset integrity check returned zero logs. Execution aborted.")
        return

    # ---------------------------------------------------------------------
    # STEP 2 — SUBSAMPLED GRID TENSOR CONSTRUCTION (16x16 Preserved)
    # ---------------------------------------------------------------------
    print("\n[STEP 2] PARSING SUBSET COORDINATES & BUILDING MASTER SLA TENSOR...")
    if dataset_outputs_exist():
        print("[+] Cache Hit: Full-grid master tensors and metadata maps already exist on disk.")
        print("[+] Skipping structural dataset rebuilding.")
    else:
        build_sla_dataset()

    # ---------------------------------------------------------------------
    # STEP 3 — LEAKAGE-FREE EXPLORATORY DATA ANALYSIS
    # ---------------------------------------------------------------------
    print("\n[STEP 3] LAUNCHING DIAGNOSTIC EXPLORATORY DATA ANALYSIS (EDA)...")
    perform_eda()

    # ---------------------------------------------------------------------
    # STEP 4 — ROBUST SCENARIO SEQUENCE ARTIFACT GENERATION
    # ---------------------------------------------------------------------
    print("\n[STEP 4] COMPILING MULTI-STEP FORECASTING SCENARIO MATRIX...")
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
    print("✓ ENTIRE MASTER DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print("\nGenerated Production-Ready Artifacts:")
    print("  -> 1. Master SLA Geospatial Matrix Tensor (.npy format)")
    print("  -> 2. Nearshore Boundary Evaluation Layer (13 Points .parquet reference)")
    print("  -> 3. Unified Global Pipeline Metadata Log (.json layout)")
    print("  -> 4. Publication-Grade Diagnostic Plots (ECDF, Boxplot, Spatial Mean, ACF)")
    print("  -> 5. 16 Systematic Scenario Subfolders containing unified compressed Train/Val/Test matrices")
    print("  -> 6. Independent MinMaxScaler parameters locked per individual scenario (.pkl configuration)")
    
    print("\nDownstream Experimental Benchmark Readiness:")
    print("  [+] Proposed Architecture   : Spatio-Temporal ConvLSTM2D Framework")
    print("  [+] Evaluated Core Baselines: Temporal LSTM | Tree-Based XGBoost | Operational Persistence")
    print("  [+] Evaluation Boundaries   : Forecast Degradation | Temporal Memory Sensitivity | Extreme-Aware Robustness")
    
    print("\nNext Immediate Deployment Step:")
    print(f"  >>> Zip and upload the directory: '{SCENARIO_OUTPUT_DIR}' to Kaggle Data Datasets.")
    print("=" * 75)


if __name__ == "__main__":
    main()