import sys
from typing import List
import gc

import pandas as pd


def optimize_floats(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    if verbose:
        print("optimizing floats")
    floats = df.select_dtypes(include=["float", "float64"]).columns.tolist()
    if len(floats) > 0:
        df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    if verbose:
        print("optimizing ints")
    ints = df.select_dtypes(include=["int64", "int32", "int"]).columns.tolist()
    if len(ints) > 0:
        df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(
    df: pd.DataFrame, datetime_features: List[str], cat_threshold: float = 0.0, verbose=False
) -> pd.DataFrame:
    if verbose:
        print("optimizing objects")
    object_cols = df.select_dtypes(include=["object"])
    for col in object_cols:
        if col not in datetime_features:
            total = max(len(df), 1)
            n_uniq = df[col].nunique()
            if n_uniq / total < cat_threshold:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize_dataframe(
    df: pd.DataFrame, datetime_features: List[str] = None, cat_threshold: float = 0.0, verbose=False
):
    datetime_features = datetime_features or []
    if verbose:
        print("Before")
        print(df.info())
    old_size = sys.getsizeof(df)
    df = optimize_objects(df, datetime_features, cat_threshold, verbose=verbose)
    df = optimize_ints(df, verbose=verbose)
    df = optimize_floats(df, verbose=verbose)
    new_size = sys.getsizeof(df)
    if verbose:
        print("optimize dataframe ({} to {}, ratio: {})".format(old_size, new_size, round(old_size / new_size, 2)))
        print(df.info())
    return df


def gc_collect_all():
    res = gc.collect(0)
    res += gc.collect(1)
    res += gc.collect(2)
    return res
