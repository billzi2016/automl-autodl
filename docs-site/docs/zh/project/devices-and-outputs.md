# 设备与输出

设备判断集中在 `utils/device_utils.py`。

## 设备检测

`detect_compute_environment()` 会检查：

- 系统里是否能找到 `nvidia-smi`。
- 当前系统是否是 macOS。
- 当前机器是否是 Apple Silicon。
- PyTorch 是否构建并可用 MPS。

优先级：

```text
CUDA -> MPS -> CPU
```

不过这个优先级主要影响 PyTorch 模型。XGBoost、LightGBM、CatBoost 的 GPU 参数只在 CUDA 分支显式启用。

## XGBoost

CUDA 环境：

```python
{
    "tree_method": "hist",
    "device": "cuda",
}
```

其他环境：

```python
{
    "tree_method": "hist",
}
```

## LightGBM

CUDA 环境：

```python
{
    "device_type": "gpu",
    "gpu_device_id": 0,
}
```

其他环境：

```python
{
    "device_type": "cpu",
}
```

## CatBoost

CUDA 环境：

```python
{
    "task_type": "GPU",
    "devices": "0",
}
```

其他环境：

```python
{
    "task_type": "CPU",
}
```

## 输出文件

AutoML：

```text
outputs/automl_grid_search_results.json
```

AutoDL：

```text
outputs/autodl_training_results.json
```

保存逻辑在 `utils/report_utils.py`：

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, output_file, ensure_ascii=False, indent=2)
```

也就是说，`outputs/` 不存在时会自动创建。

## 摘要打印

训练结束后会打印：

```text
Model search summary:
- model_name: auc=..., f1=..., accuracy=..., params=...
```

如果模型因为依赖缺失被跳过，会打印：

```text
- xgboost_model: skipped, reason=xgboost is not installed.
```
