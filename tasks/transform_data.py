from crewai import Task
from agents.preprocessor import preprocessor
from utils.preprocessing import normalize

def transform_dataset(df, method="standard"):
    """Transform dataset by normalizing numeric columns."""
    df = normalize(df, method=method)
    return df


transform_data = Task(
    description="Transform dataset by normalizing numeric columns.",
    agent=preprocessor,
    expected_output="Normalized dataset ready for analysis or ML."
)
