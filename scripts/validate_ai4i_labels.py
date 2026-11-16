"""Audit AI4I Machine failure versus failure-mode flags without editing raw data.

Usage: python scripts/validate_ai4i_labels.py --input data/raw/ai4i2020.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

TARGET = "Machine failure"
MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(input_path: Path, output_dir: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Data file not found: {input_path}")
    before_hash = sha256(input_path)
    total = 0
    target_counts = Counter()
    mode_counts = Counter()
    discordance_counts = Counter()
    mismatches = []
    overlap_count = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {TARGET, *MODES}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}; observed: {columns}")
        id_col = next((col for col in ("UDI", "UID") if col in columns), None)
        for row_num, row in enumerate(reader, start=2):
            total += 1
            values = {name: row.get(name, "").strip() for name in (TARGET, *MODES)}
            invalid = [key for key, value in values.items() if value not in ("0", "1")]
            if invalid:
                raise ValueError(f"Row {row_num}: non-binary or empty values: {invalid}")
            target = int(values[TARGET])
            active = [m for m in MODES if values[m] == "1"]
            target_counts[str(target)] += 1
            for mode in active:
                mode_counts[mode] += 1
            if len(active) >= 2:
                overlap_count += 1
            any_mode = bool(active)
            if any_mode != bool(target):
                kind = "mode_flag_but_normal" if any_mode else "failure_without_mode_flag"
                discordance_counts[kind] += 1
                mismatches.append({
                    "csv_row": row_num,
                    "UDI_or_UID": row[id_col] if id_col else "",
                    "Product ID": row.get("Product ID", ""),
                    "Type": row.get("Type", ""),
                    TARGET: target,
                    **{mode: int(values[mode]) for mode in MODES},
                    "active_mode_count": len(active),
                    "discrepancy": kind,
                })

    after_hash = sha256(input_path)
    if before_hash != after_hash:
        raise RuntimeError("Input data changed during validation; no outputs were written")
    if total == 0:
        raise ValueError("Input CSV is empty")

    summary = {
        "source_file": str(input_path),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_unchanged": True,
        "total_rows": total,
        "target_counts": dict(sorted(target_counts.items())),
        "mode_counts": {mode: mode_counts[mode] for mode in MODES},
        "rows_with_multiple_mode_flags": overlap_count,
        "discrepancy_counts": {
            "mode_flag_but_normal": discordance_counts["mode_flag_but_normal"],
            "failure_without_mode_flag": discordance_counts["failure_without_mode_flag"],
        },
        "discrepancy_rows": len(mismatches),
        "discrepancy_rate": len(mismatches) / total,
        "interpretation": (
            "Descriptive label audit only. Do not edit/drop rows automatically. "
            "Compare dataset revision, source documentation, and modeling policy separately."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "label_discrepancies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "csv_row", "UDI_or_UID", "Product ID", "Type", TARGET,
            *MODES, "active_mode_count", "discrepancy",
        ])
        writer.writeheader()
        writer.writerows(mismatches)
    (output_dir / "label_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="AI4I CSV path")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args()
    summary = audit(args.input, args.output)
    print(json.dumps({
        "rows": summary["total_rows"],
        "discrepancies": summary["discrepancy_rows"],
        "discrepancy_counts": summary["discrepancy_counts"],
        "mode_counts": summary["mode_counts"],
        "source_unchanged": summary["source_unchanged"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
