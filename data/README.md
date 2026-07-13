# Data

This folder contains the dataset configuration and mapping details required for the spatio-temporal SLA forecasting pipeline. The data represents decadal gridded observations along the northern coast of Java.

## Datasets

### Sea Level Anomaly Semarang

- **Link:** [Kaggle: sea-level-anomaly-semarang](https://www.kaggle.com/datasets/muhffikkri/sea-level-anomaly-semarang)
- **Description:** This dataset serves as the core physical observation repository for the spatiotemporal model. It encompasses 10 years of daily reprocessed Level-4 Sea Level Anomaly (SLA) gridded satellite observations (January 2016 – August 2025) sourced from the Copernicus Marine Service (CMEMS). It is bounded globally by a $16 \times 16$ spatial grid resolution ($0.125^{\circ} \times 0.125^{\circ}$ grid cells) spanning $109.5^{\circ}\text{E}-111.5^{\circ}\text{E}$ and $6.0^{\circ}\text{S}-8.0^{\circ}\text{S}$ surrounding the Semarang coastal region.

The repository structure contains:

- `scenarios/` — Folder containing the pre-processed sliding-window data bundles (e.g., `.npz` compressed tensors for training, validation, and testing) categorized by different input histories ($T$) and forecast horizons ($H$).

- `reference/coastal_points.parquet` — A reference file detailing the precise geographic row and column grid indices (`lat_idx`, `lon_idx`) mapping out the 13 active coastal cells under study.

- `metadata/eda_summary.json` — Pre-calculated historical percentile metrics ($P_{95}$) extracted strictly from the training set, establishing the exact baseline limits for the extreme-event loss engine.

- `scaler.pkl` — Pre-saved serialization of the MinMaxScaler fitted exclusively on the training array to prevent spatial data leakage during evaluation.

This dataset is used by the preprocessing framework (`src/preprocess/`) to slice continuous matrices and feed the downstream deep learning pipeline (`src/notebook/training_pipeline.ipynb`).
