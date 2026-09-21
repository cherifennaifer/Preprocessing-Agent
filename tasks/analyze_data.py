from crewai import Task
from agents.preprocessor import preprocessor
import pandas as pd

def analyze_dataset(df: pd.DataFrame):
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": df.duplicated().sum(),
        "statistics": df.describe(include="all").to_dict()
    }
    return summary

analyze_data = Task(
    description="Analyze dataset and explain duplicates, missing values, and statistics.",
    agent=preprocessor,
    expected_output="Detailed feedback with advice on preprocessing steps."
)
