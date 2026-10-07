"""Load and validate user-provided input path tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InputRecord:
    input_path: Path
    output_id: str
    row_number: int
    extra: dict[str, str] = field(default_factory=dict)


def strip_nifti_suffix(path: Path) -> str:
    """Return a NIfTI filename without .nii or .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def validate_output_id(output_id: str, row_number: int) -> str:
    clean = output_id.strip()
    if not clean:
        raise ValueError(f"Row {row_number}: output_id is empty.")
    if "/" in clean or "\\" in clean:
        raise ValueError(
            f"Row {row_number}: output_id must be a filename prefix, not a path. Got: {output_id!r}"
        )
    return clean


def load_input_paths(path: str | Path) -> list[InputRecord]:
    """Read an input path file and return validated records.

    Supported formats:
    - .csv with input_path and optional output_id columns
    - .tsv with input_path and optional output_id columns
    - plain text with one input path per line

    Plain text files may contain blank lines and # comments.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path file is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _load_delimited_input_paths(path, delimiter=",", format_name="CSV")
    elif suffix == ".tsv":
        records = _load_delimited_input_paths(path, delimiter="\t", format_name="TSV")
    else:
        records = _load_text_input_paths(path)

    check_unique_output_ids(records)
    return records


def _make_record(raw_input: str, raw_output_id: str, row_number: int, extra: dict[str, str] | None = None) -> InputRecord:
    if not raw_input:
        raise ValueError(f"Row {row_number}: input_path is empty.")

    input_path = Path(raw_input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Row {row_number}: input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Row {row_number}: input_path is not a file: {input_path}")

    output_id = validate_output_id(raw_output_id or strip_nifti_suffix(input_path), row_number)
    return InputRecord(input_path=input_path, output_id=output_id, row_number=row_number, extra=extra or {})


def _load_delimited_input_paths(path: Path, delimiter: str, format_name: str) -> list[InputRecord]:
    records: list[InputRecord] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Input path {format_name} has no header row: {path}")
        if "input_path" not in reader.fieldnames:
            raise ValueError(f"Input path {format_name} must contain an input_path column.")

        for row_number, row in enumerate(reader, start=2):
            raw_input = (row.get("input_path") or "").strip()
            raw_output_id = (row.get("output_id") or "").strip()
            extra = {k: v for k, v in row.items() if k not in {"input_path", "output_id"}}
            records.append(_make_record(raw_input, raw_output_id, row_number, extra=extra))

    return records


def _load_text_input_paths(path: Path) -> list[InputRecord]:
    records: list[InputRecord] = []
    with path.open("r") as f:
        for row_number, line in enumerate(f, start=1):
            raw_input = line.strip()
            if not raw_input or raw_input.startswith("#"):
                continue
            records.append(_make_record(raw_input, raw_output_id="", row_number=row_number))

    if not records:
        raise ValueError(f"Input path text file contains no usable paths: {path}")

    return records


def check_unique_output_ids(records: list[InputRecord]) -> None:
    by_id: dict[str, list[InputRecord]] = {}
    for record in records:
        by_id.setdefault(record.output_id, []).append(record)

    duplicates = {k: v for k, v in by_id.items() if len(v) > 1}
    if not duplicates:
        return

    parts = ["Multiple input files would produce the same output filename prefix."]
    for output_id, dup_records in duplicates.items():
        parts.append("")
        parts.append(f"Output prefix: {output_id}")
        parts.append("Conflicting input files:")
        for rec in dup_records:
            parts.append(f"  row {rec.row_number}: {rec.input_path}")

    parts.append("")
    parts.append("By default, csfseg uses the input filename without .nii/.nii.gz as the output prefix.")
    parts.append("To fix this, use a CSV or TSV input table and add an output_id column.")
    parts.append("")
    parts.append("Example:")
    parts.append("  input_path,output_id")
    parts.append("  /data/siteA/sub-001/func/rest_bold.nii.gz,siteA_sub-001_rest_bold")
    parts.append("  /data/siteB/sub-002/func/rest_bold.nii.gz,siteB_sub-002_rest_bold")
    raise ValueError("\n".join(parts))


def write_input_paths_used(records: list[InputRecord], out_path: str | Path) -> Path:
    """Write the resolved input table used for a run."""
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extra_keys: list[str] = []
    for record in records:
        for key in record.extra:
            if key not in extra_keys:
                extra_keys.append(key)

    fieldnames = ["input_path", "output_id", *extra_keys]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "input_path": str(record.input_path),
                "output_id": record.output_id,
            }
            row.update(record.extra)
            writer.writerow(row)

    return out_path
