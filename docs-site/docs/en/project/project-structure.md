# Project structure

```text
automl-autodl/
├── automl/
├── autodl/
├── data/
├── docs-site/
├── utils/
├── config.py
├── preprocess.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
└── README_CN.md
```

`main.py` is the AutoML entrypoint.

`autodl/train.py` is the AutoDL entrypoint.

`preprocess.py` builds the shared feature matrix.

`config.py` owns data paths, feature lists, model lists, scoring settings, and training parameters.

`utils/train_utils.py` contains the shared grid search wrapper and PyTorch training helpers.

`utils/device_utils.py` handles CUDA, MPS, CPU, and library-specific runtime parameters.

`utils/report_utils.py` prints summaries and writes JSON output.
