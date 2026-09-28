"""Robust parser for MSI Afterburner Hardware monitoring log files.

The MSI Afterburner text log is not a normal CSV file. It contains record types:
    00 - log metadata
    01 - monitored hardware/device
    02 - metric names
    03 - metric metadata (units/ranges/format)
    80 - telemetry samples

This parser uses the 02/03/80 records instead of hard-coded row numbers or
column positions. Missing optional metrics are therefore normal and do not
cause parsing failures.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricMetadata:
    """Metadata reported by an MSI Afterburner 03 record."""

    index: int
    name: str
    unit: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    format_string: Optional[str] = None


@dataclass
class AfterburnerLog:
    """Parsed MSI Afterburner log and its metadata."""

    dataframe: pd.DataFrame
    metrics: List[str]
    metric_metadata: Dict[str, MetricMetadata] = field(default_factory=dict)
    hardware: List[str] = field(default_factory=list)
    log_started_at: Optional[pd.Timestamp] = None
    source_path: Optional[Path] = None
    record_count: int = 0

    @property
    def available_metrics(self) -> set[str]:
        return set(self.metrics)

    def has_metric(self, name: str) -> bool:
        return name in self.dataframe.columns

    def missing_metrics(self, names: Sequence[str]) -> List[str]:
        return [name for name in names if not self.has_metric(name)]


class AfterburnerLogParser:
    """Parse MSI Afterburner monitoring logs without assuming optional columns."""

    DATA_RECORD = "80"
    METRIC_RECORD = "02"
    METADATA_RECORD = "03"
    HARDWARE_RECORD = "01"

    def __init__(self, path: str | Path, encoding: str = "utf-8-sig") -> None:
        self.path = Path(path)
        self.encoding = encoding

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip().strip("\ufeff")

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        value = value.strip()
        if not value or value.upper() in {"N/A", "NA", "NULL", "NONE", "-"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _unique_names(names: Sequence[str]) -> List[str]:
        """Preserve names while making duplicate metric names unambiguous."""
        counts: Dict[str, int] = {}
        result: List[str] = []
        for name in names:
            base = name or "metric"
            count = counts.get(base, 0)
            counts[base] = count + 1
            result.append(base if count == 0 else f"{base}_{count + 1}")
        return result

    def parse(self) -> AfterburnerLog:
        if not self.path.exists():
            raise FileNotFoundError(f"Afterburner log not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Afterburner log path is not a file: {self.path}")

        metric_names: List[str] = []
        metric_metadata_by_index: Dict[int, MetricMetadata] = {}
        hardware: List[str] = []
        samples: List[List[str]] = []
        log_started_at: Optional[pd.Timestamp] = None

        with self.path.open("r", encoding=self.encoding, errors="replace", newline="") as handle:
            reader = csv.reader(handle, skipinitialspace=False)

            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                record_type = self._clean(row[0])
                fields = [self._clean(value) for value in row]

                if len(fields) > 1 and log_started_at is None:
                    log_started_at = self._parse_timestamp(fields[1])

                if record_type == self.HARDWARE_RECORD:
                    if len(fields) > 2 and fields[2]:
                        hardware.append(fields[2])

                elif record_type == self.METRIC_RECORD:
                    # Record 02: type, timestamp, metric0, metric1, ...
                    raw_names = fields[2:]
                    while raw_names and not raw_names[-1]:
                        raw_names.pop()
                    metric_names = self._unique_names(raw_names)

                elif record_type == self.METADATA_RECORD:
                    # Record 03: type, timestamp, metric index, unit, min, max, step, ..., format
                    self._parse_metric_metadata(fields, metric_names, metric_metadata_by_index)

                elif record_type == self.DATA_RECORD:
                    # Record 80: type, timestamp, value0, value1, ...
                    if len(fields) < 2:
                        logger.warning("Ignoring malformed record 80 on line %d", line_number)
                        continue
                    samples.append(fields[1:])

        if not metric_names:
            raise ValueError(
                "No MSI Afterburner metric header (record type 02) was found. "
                "The file may not be an Afterburner hardware monitoring log."
            )

        if not samples:
            raise ValueError("No MSI Afterburner telemetry records (record type 80) were found.")

        dataframe = self._build_dataframe(samples, metric_names)

        metadata: Dict[str, MetricMetadata] = {}
        for index, item in metric_metadata_by_index.items():
            if 0 <= index < len(metric_names):
                # Use the normalized/unique metric name from record 02.
                item.name = metric_names[index]
                metadata[item.name] = item

        # Timestamp is always represented consistently for downstream analysis.
        dataframe["format_time_aft"] = self._parse_timestamp_series(dataframe["timestamp"])
        dataframe.drop(columns=["timestamp"], inplace=True)
        dataframe = dataframe.sort_values("format_time_aft", kind="stable").reset_index(drop=True)

        # Convert all telemetry metrics to numeric. Invalid values become NaN.
        for metric in metric_names:
            dataframe[metric] = pd.to_numeric(dataframe[metric], errors="coerce")

        return AfterburnerLog(
            dataframe=dataframe,
            metrics=metric_names,
            metric_metadata=metadata,
            hardware=hardware,
            log_started_at=log_started_at,
            source_path=self.path,
            record_count=len(dataframe),
        )

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[pd.Timestamp]:
        value = value.strip().replace("-", "/")
        if not value:
            return None
        return pd.to_datetime(value, dayfirst=True, errors="coerce")

    @classmethod
    def _parse_timestamp_series(cls, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.replace("-", "/", regex=False)
        return pd.to_datetime(cleaned, dayfirst=True, errors="coerce")

    @classmethod
    def _build_dataframe(cls, samples: List[List[str]], metric_names: List[str]) -> pd.DataFrame:
        expected_length = len(metric_names) + 1  # timestamp + metrics
        normalized: List[List[str]] = []

        for sample in samples:
            values = list(sample)
            if len(values) < expected_length:
                values.extend([""] * (expected_length - len(values)))
            elif len(values) > expected_length:
                logger.warning(
                    "Telemetry row has %d extra fields; ignoring them.",
                    len(values) - expected_length,
                )
            normalized.append(values[:expected_length])

        columns = ["timestamp", *metric_names]
        return pd.DataFrame(normalized, columns=columns)

    @classmethod
    def _parse_metric_metadata(
        cls,
        fields: List[str],
        metric_names: List[str],
        destination: Dict[int, MetricMetadata],
    ) -> None:
        if len(fields) < 3:
            return

        try:
            index = int(fields[2])
        except ValueError:
            return

        if index < 0 or index >= len(metric_names):
            return

        unit = fields[3] if len(fields) > 3 and fields[3] else None
        minimum = cls._to_float(fields[4]) if len(fields) > 4 else None
        maximum = cls._to_float(fields[5]) if len(fields) > 5 else None
        step = cls._to_float(fields[6]) if len(fields) > 6 else None
        format_string = fields[-1] if fields and fields[-1] else None

        destination[index] = MetricMetadata(
            index=index,
            name=metric_names[index],
            unit=unit,
            minimum=minimum,
            maximum=maximum,
            step=step,
            format_string=format_string,
        )


def load_afterburner_log(path: str | Path) -> AfterburnerLog:
    """Convenience function for importing an MSI Afterburner log."""
    return AfterburnerLogParser(path).parse()
