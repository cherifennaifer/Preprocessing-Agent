from crewai import Task
from agents.preprocessor import preprocessor
from utils.preprocessing import fill_missing, remove_duplicates

def clean_dataset(df, missing_strategy="mean", remove_dupes=True):
    if remove_dupes:
        df = remove_duplicates(df)
    df = fill_missing(df, strategy=missing_strategy)
    return df

clean_data = Task(
    description="Clean dataset by removing duplicates and handling missing values.",
    agent=preprocessor,
    expected_output="Cleaned dataset after applying chosen strategies."
)
