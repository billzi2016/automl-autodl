# Devices and outputs

Device selection is centralized in `utils/device_utils.py`.

## Detection

`detect_compute_environment()` checks:

- whether `nvidia-smi` exists
- whether the system is macOS
- whether the machine is Apple Silicon
- whether PyTorch MPS is built and available

Preference:

```text
CUDA -> MPS -> CPU
```

This mainly affects PyTorch models. XGBoost, LightGBM, and CatBoost only enable GPU parameters on CUDA branches.

## Output files

AutoML:

```text
outputs/automl_grid_search_results.json
```

AutoDL:

```text
outputs/autodl_training_results.json
```

`save_results()` creates the output directory if needed and writes UTF-8 JSON with indentation.
