"""Data ingestion layer — swappable loaders for CSV, API, etc."""

from .csv_loader import CSVDataLoader  # noqa: F401

__all__ = ["CSVDataLoader"]
