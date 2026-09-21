import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file, file_type):
    if file_type == "csv":
        return pd.read_csv(file)
    elif file_type == "xlsx":
        return pd.read_excel(file)
    elif file_type == "json":
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file type")

def fill_missing(df, strategy="mean", columns=None):
    columns = columns or list(df.select_dtypes(include=["float64", "int64"]).columns)
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def normalize(df, method="standard", columns=None):
    numeric_cols = (
        list(df.select_dtypes(include=["number"]).columns)
        if columns is None
        else [column for column in columns if column in df.columns]
    )
    if len(numeric_cols) == 0:
        return df
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization method")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
