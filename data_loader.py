from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_FILES = {
    "geo": "GeoLocations.xlsx",
    "demand": "DemandCluster.xlsx",
    "distance": "DistanceCluster.xlsx",
    "cost": "CostCluster.xlsx",
    "capacity": "CapacityClustered.xlsx",
    "time": "Time.xlsx",
    "fixed_cost": "CostsMWC-Clustered.xlsx",
}


@dataclass
class WorkbookBundle:
    raw_sheets: dict[str, pd.DataFrame]
    cleaned: dict[str, Any]


def _dataset_root(base_path: str | Path | None = None) -> Path:
    if base_path is None:
        return Path(__file__).resolve().parent / "dataset"
    return Path(base_path)


def _drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def _as_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _metrics_before_blank_block(df: pd.DataFrame) -> pd.DataFrame:
    working = _drop_empty(df)
    blank_rows = working.index[working.isna().all(axis=1)].tolist()
    if blank_rows:
        working = working.iloc[: blank_rows[0]]
    working = _drop_empty(working)
    return working


def _clean_geolocations(path: Path) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.rename(columns={frame.columns[0]: "lat", frame.columns[1]: "lon"})
        frame = frame[["lat", "lon"]].dropna(how="all")
        frame["region"] = sheet_name
        frame["point_id"] = [f"P{i + 1}" for i in range(len(frame))]
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["outlier"] = ~(
            frame["lat"].between(39, 42, inclusive="both")
            & frame["lon"].between(39, 42, inclusive="both")
        )
        cleaned[sheet_name] = frame.reset_index(drop=True)
    return cleaned


def _clean_demand(path: Path) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        if sheet_name == "Gercek Data":
            frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
            cleaned[sheet_name] = _drop_empty(frame)
            continue

        frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
        frame = _metrics_before_blank_block(frame)
        frame = frame.iloc[:, : frame.notna().any(axis=0).sum()]
        metric_names = frame.iloc[:, 0].astype(str).str.strip().tolist()
        numeric = _as_numeric_frame(frame.iloc[:, 1:])
        numeric = numeric.loc[:, numeric.notna().any(axis=0)]
        numeric.columns = [f"Cluster {i + 1}" for i in range(numeric.shape[1])]
        wide = numeric.copy()
        wide.index = metric_names
        wide.index.name = "demand_type"
        melted = wide.reset_index().melt(id_vars="demand_type", var_name="cluster", value_name="demand")
        melted["region"] = sheet_name.replace("Demand ", "")
        cleaned[sheet_name] = melted.dropna(subset=["demand"]).reset_index(drop=True)
    return cleaned


def _clean_capacity(path: Path) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
        frame = _drop_empty(frame)
        records: list[dict[str, Any]] = []
        idx = 0
        while idx < len(frame):
            label = frame.iloc[idx, 0]
            if pd.isna(label):
                idx += 1
                continue
            if isinstance(label, str) and label.startswith("P="):
                configuration = label
                metric_rows = frame.iloc[idx + 1 : idx + 6].copy()
                metric_rows = metric_rows.dropna(axis=1, how="all")
                if metric_rows.empty:
                    idx += 1
                    continue
                point_labels = [f"Point {i + 1}" for i in range(metric_rows.shape[1] - 1)]
                for _, row in metric_rows.iterrows():
                    metric = str(row.iloc[0]).strip()
                    values = pd.to_numeric(row.iloc[1:], errors="coerce").tolist()
                    for point_label, value in zip(point_labels, values, strict=False):
                        records.append(
                            {
                                "region": sheet_name.replace("Capacity ", ""),
                                "configuration": configuration,
                                "metric": metric,
                                "point": point_label,
                                "value": value,
                            }
                        )
                idx += 6
            else:
                idx += 1
        cleaned[sheet_name] = pd.DataFrame(records)
    return cleaned


def _clean_time(path: Path) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        first_col = frame.columns[0]
        frame = frame.rename(columns={first_col: "source_zone"})
        frame = frame.dropna(axis=1, how="all")
        frame["source_zone"] = frame["source_zone"].astype(str).str.strip()
        frame = frame[frame["source_zone"].str.lower() != "nan"].copy()
        long_frame = frame.melt(id_vars="source_zone", var_name="pharmacy_point", value_name="delivery_time")
        long_frame["region"] = sheet_name.replace("Time ", "")
        cleaned[sheet_name] = long_frame.dropna(subset=["delivery_time"]).reset_index(drop=True)
    return cleaned


def _clean_matrix_workbook(path: Path, prefix: str) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.rename(columns={frame.columns[0]: "cluster"})
        frame = frame.dropna(axis=1, how="all")
        long_frame = frame.melt(id_vars="cluster", var_name="mode", value_name="value")
        long_frame["region"] = sheet_name.replace(prefix, "").strip()
        cleaned[sheet_name] = long_frame.dropna(subset=["value"]).reset_index(drop=True)
    return cleaned


def _clean_fixed_cost(path: Path) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for sheet_name in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        first_col = frame.columns[0]
        frame = frame.rename(columns={first_col: "warehouse"})
        frame = frame.dropna(axis=1, how="all")
        long_frame = frame.melt(id_vars="warehouse", var_name="cost_type", value_name="fixed_cost")
        long_frame["region"] = sheet_name.replace("Costs ", "")
        cleaned[sheet_name] = long_frame.dropna(subset=["fixed_cost"]).reset_index(drop=True)
    return cleaned


def load_all_workbooks(base_path: str | Path | None = None) -> dict[str, WorkbookBundle]:
    root = _dataset_root(base_path)
    cleaned_workbooks = {
        "geo": _clean_geolocations(root / DATASET_FILES["geo"]),
        "demand": _clean_demand(root / DATASET_FILES["demand"]),
        "distance": _clean_matrix_workbook(root / DATASET_FILES["distance"], "Distance "),
        "cost": _clean_matrix_workbook(root / DATASET_FILES["cost"], "Cost "),
        "capacity": _clean_capacity(root / DATASET_FILES["capacity"]),
        "time": _clean_time(root / DATASET_FILES["time"]),
        "fixed_cost": _clean_fixed_cost(root / DATASET_FILES["fixed_cost"]),
    }
    bundles: dict[str, WorkbookBundle] = {}
    for key, file_name in DATASET_FILES.items():
        workbook_path = root / file_name
        xls = pd.ExcelFile(workbook_path)
        raw_sheets = {sheet_name: pd.read_excel(workbook_path, sheet_name=sheet_name, header=None) for sheet_name in xls.sheet_names}
        bundles[key] = WorkbookBundle(raw_sheets=raw_sheets, cleaned=cleaned_workbooks[key])
    return bundles


def collect_quality_summary(bundles: dict[str, WorkbookBundle]) -> dict[str, Any]:
    report: dict[str, Any] = {"workbooks": {}, "geo_outliers": {}, "zero_demand_entries": {}}
    for workbook_name, bundle in bundles.items():
        workbook_report: dict[str, Any] = {}
        for sheet_name, raw_frame in bundle.raw_sheets.items():
            numeric_frame = raw_frame.select_dtypes(include=[np.number])
            workbook_report[sheet_name] = {
                "shape": raw_frame.shape,
                "missing_values": int(raw_frame.isna().sum().sum()),
                "duplicate_rows": int(raw_frame.duplicated().sum()),
                "zero_cells": int((numeric_frame == 0).sum().sum()) if not numeric_frame.empty else 0,
            }
        report["workbooks"][workbook_name] = workbook_report
    geo_bundle = bundles["geo"].cleaned
    report["geo_outliers"] = {
        sheet_name: int(frame["outlier"].sum()) for sheet_name, frame in geo_bundle.items()
    }
    demand_zeros: dict[str, int] = {}
    for sheet_name, frame in bundles["demand"].cleaned.items():
        if isinstance(frame, pd.DataFrame) and "demand" in frame.columns:
            demand_zeros[sheet_name] = int((frame["demand"].fillna(0) == 0).sum())
    report["zero_demand_entries"] = demand_zeros
    return report


def print_quality_summary(report: dict[str, Any]) -> None:
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    for workbook_name, sheets in report["workbooks"].items():
        print(f"Workbook: {workbook_name}")
        for sheet_name, stats in sheets.items():
            print(
                f"  {sheet_name}: shape={stats['shape']}, missing={stats['missing_values']}, "
                f"duplicates={stats['duplicate_rows']}, zero_cells={stats['zero_cells']}"
            )
    print("Geo outliers (lat/lon outside 39-42):")
    for sheet_name, count in report["geo_outliers"].items():
        print(f"  {sheet_name}: {count}")
    print("Zero-demand entries:")
    for sheet_name, count in report["zero_demand_entries"].items():
        print(f"  {sheet_name}: {count}")
    print("=" * 80)


def load_and_validate(base_path: str | Path | None = None) -> tuple[dict[str, WorkbookBundle], dict[str, Any]]:
    bundles = load_all_workbooks(base_path)
    report = collect_quality_summary(bundles)
    return bundles, report
