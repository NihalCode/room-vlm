# Room VLM Evaluation Report Template
#
# This file is a tracked template. Generated reports with real metrics belong
# under gitignored outputs/evaluation/report.md after you run evaluate.py.

## Dataset
_Populate from data/metadata/dataset_stats.json after prepare_nyu.py._

## Model
Base: Qwen/Qwen3-VL-4B-Instruct  
Adapter: _path/name after training_

## Training Configuration
_See outputs/<run_name>/config.yaml_

## Zero-Shot Baseline
_See outputs/baseline/metrics.json — do not invent numbers._

## Fine-Tuned Results
_See outputs/evaluation/metrics.json — do not invent numbers._

## Per-Class Results
_Filled by evaluate.py_

## Confusion Matrix
_See confusion_matrix.csv / .png_

## Calibration
Candidate likelihoods are not necessarily calibrated probabilities.

## Video Evaluation
_Optional_

## Error Analysis
See errors.csv / errors.html (hypotheses only).

## Limitations
See README.

## Reproducibility
seed=42 where configured; full GPU bit-exact reproducibility is not guaranteed.
