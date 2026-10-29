"""
Base Data Loader
================
Abstract interface that every data-source loader must implement.
This ensures the rest of the pipeline is decoupled from the actual
data source (CSV today, financial API tomorrow).
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loaders.

    Any concrete loader (CSV, API, database, …) must implement
    ``load_data()`` and return a DataFrame with the standard columns:

        timestamp | commodity | open | high | low | close | volume
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data and return a standardized Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            timestamp (datetime64), commodity (str),
            open (float), high (float), low (float),
            close (float), volume (float/int).
        """
        pass
