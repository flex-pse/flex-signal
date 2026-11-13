"""File input and output: Parquet (lossless, with flags and metadata) and CSV."""

from flexsignal.io.files import (
    METADATA_KEY,
    read_csv,
    read_parquet,
    write_csv,
    write_parquet,
)

__all__ = ["METADATA_KEY", "read_csv", "read_parquet", "write_csv", "write_parquet"]
