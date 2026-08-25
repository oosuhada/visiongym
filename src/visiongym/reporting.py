from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _accuracy_frame(section: dict[str, Any], label: str) -> pd.DataFrame:
    rows = [
        {label: key, "accuracy": values.get("accuracy"), "samples": values.get("samples")}
        for key, values in section.items()
    ]
    return pd.DataFrame(rows)


def create_report(metrics_path: str | Path, output_dir: str | Path) -> list[Path]:
    metrics = _load_metrics(metrics_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    task_frame = _accuracy_frame(metrics.get("by_task", {}), "task")
    if not task_frame.empty:
        task_csv = output / "task_accuracy.csv"
        task_frame.to_csv(task_csv, index=False)
        created.append(task_csv)
        axis = task_frame.sort_values("accuracy").plot.barh(x="task", y="accuracy", legend=False, figsize=(8, 5))
        axis.set_xlim(0, 1)
        axis.set_title("Accuracy by task")
        figure = axis.get_figure()
        figure.tight_layout()
        chart = output / "task_accuracy.png"
        figure.savefig(chart, dpi=160)
        plt.close(figure)
        created.append(chart)

    ood_frame = _accuracy_frame(metrics.get("by_ood_type", {}), "ood_type")
    if not ood_frame.empty:
        ood_csv = output / "ood_accuracy.csv"
        ood_frame.to_csv(ood_csv, index=False)
        created.append(ood_csv)
        axis = ood_frame.sort_values("accuracy").plot.barh(x="ood_type", y="accuracy", legend=False, figsize=(7, 4))
        axis.set_xlim(0, 1)
        axis.set_title("Accuracy by OOD condition")
        figure = axis.get_figure()
        figure.tight_layout()
        chart = output / "ood_accuracy.png"
        figure.savefig(chart, dpi=160)
        plt.close(figure)
        created.append(chart)

    summary_path = output / "summary.md"
    summary_lines = [
        "# VisionGym Evaluation Summary",
        "",
        f"- Model: `{metrics.get('model') or 'unknown'}`",
        f"- Prompt mode: `{metrics.get('prompt_mode') or 'unknown'}`",
        f"- Samples: {metrics.get('samples', 0)}",
        f"- Overall accuracy: {_format_pct(metrics.get('overall_accuracy'))}",
        f"- ID accuracy: {_format_pct(metrics.get('id_accuracy'))}",
        f"- OOD accuracy: {_format_pct(metrics.get('ood_accuracy'))}",
        f"- OOD gap: {_format_pct(metrics.get('ood_gap'))}",
        f"- Invalid output rate: {_format_pct(metrics.get('invalid_output_rate'))}",
        "",
        "## Error distribution",
        "",
    ]
    error_distribution = metrics.get("error_distribution", {})
    if error_distribution:
        summary_lines.extend(f"- {name}: {count}" for name, count in error_distribution.items())
    else:
        summary_lines.append("- No errors recorded.")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    created.append(summary_path)
    return created


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"

