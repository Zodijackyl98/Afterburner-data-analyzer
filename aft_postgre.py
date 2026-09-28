"""Import an MSI Afterburner log into PostgreSQL.

The importer intentionally does not require every possible Afterburner metric.
The table contains whatever telemetry the source log actually provides.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from sqlalchemy import create_engine

from afterburner_parser import load_afterburner_log


PATH = Path(r"C:\Python_Works\py\afterburner-data-analyzer\examples\texts\HMW_MHW_2.txt")
TABLE_NAME = "hmw_mhw_2"

DB_USER = os.getenv("AFTERBURNER_DB_USER")
DB_PASSWORD = os.getenv("AFTERBURNER_DB_PASSWORD")
DB_NAME = os.getenv("AFTERBURNER_DB_NAME")
DB_HOST = os.getenv("AFTERBURNER_DB_HOST")
DB_PORT = os.getenv("AFTERBURNER_DB_PORT")


def safe_identifier(value: str) -> str:
    """Allow only PostgreSQL identifier characters for generated table names."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe PostgreSQL table name: {value!r}")
    return value


def main() -> None:
    table_name = safe_identifier(TABLE_NAME)

    parsed = load_afterburner_log(PATH)
    df = parsed.dataframe.copy()

    # Keep only one row per timestamp. The parser itself does not silently
    # discard duplicates because that is a data-policy decision, not parsing.
    df = df.drop_duplicates(subset=["format_time_aft"], keep="first")

    # PostgreSQL/Superset-friendly names. Keep the original MSI names in the
    # source CSV if exact source names are required later.
    df = df.rename(columns={"format_time_aft": "timestamp"})
    df.insert(0, "source_file", PATH.name)

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # 'replace' is retained here for compatibility with the old importer.
    # Once sessions are introduced, switch this to append and use a session_id.
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)

    print(f"Imported {len(df):,} telemetry rows into {table_name!r}.")
    print(f"Detected hardware: {', '.join(parsed.hardware) or 'unknown'}")
    print("Available metrics:")
    for metric in parsed.metrics:
        metadata = parsed.metric_metadata.get(metric)
        unit = f" [{metadata.unit}]" if metadata and metadata.unit else ""
        print(f"  - {metric}{unit}")


if __name__ == "__main__":
    main()
