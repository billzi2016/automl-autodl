from __future__ import annotations

import json
from pathlib import Path


def print_result_summary(results):
    """Print a concise summary for all model runs."""
    # 这里统一打印核心结果，方便后续查看和整理实验输出。
    print("\nModel search summary:")
    for result in results:
        model_name = result["model_name"]
        status = result["status"]

        if status == "completed":
            metrics = result["cv_metrics"]
            print(
                f"- {model_name}: auc={metrics['roc_auc']:.6f}, "
                f"f1={metrics['f1']:.6f}, "
                f"accuracy={metrics['accuracy']:.6f}, "
                f"params={result['best_params']}"
            )
        else:
            print(f"- {model_name}: skipped, reason={result['reason']}")


def save_results(results, output_path: str | Path) -> None:
    """Save model search results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)
