"""
    Reformat dataframes or any other data structures.
"""

from typing import List
import pandas as pd


def rearrange_cols_first(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rearrange the dataframe's columns,
    by setting `cols` as the first columns"""
    return df[cols + [c for c in df.columns if c not in cols]]


def rearrange_cols_last(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rearrange the dataframe's columns,
    by setting `cols` as the last columns"""
    return df[[c for c in df.columns if c not in cols] + cols]
