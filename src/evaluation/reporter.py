"""
Evaluation Report Generator (Markdown & CSV).
PS 26052 — Adaptive Defence ANC.
"""

from typing import List, Dict, Any
from pathlib import Path
import csv


class EvaluationReporter:
    """Formats benchmark runs into publication-ready tables for jury presentation."""

    @staticmethod
    def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
        """Generate formatted GitHub Markdown comparison table."""
        lines = [
            "| Condition / Model | Input SNR (dB) | Output SNR (dB) | SI-SNR (dB) | STOI | PESQ | Target Compliant |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for r in results:
            passed = "✅ PASS" if r.get("all_targets_passed", False) else "⚠️ TESTING"
            lines.append(
                f"| **{r.get('name', 'N/A')}** | {r.get('input_snr', 'N/A')} | "
                f"{r.get('snr_db', 'N/A')} | {r.get('si_snr_db', 'N/A')} | "
                f"{r.get('stoi', 'N/A')} | {r.get('pesq', 'N/A')} | {passed} |"
            )
        return "\n".join(lines)

    @staticmethod
    def save_csv(output_path: str, results: List[Dict[str, Any]]) -> None:
        """Export results to CSV."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if results:
            keys = list(results[0].keys())
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
