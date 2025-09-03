import pandas as pd

class AnimeDataLoader:
    def __init__(self, original_csv : str, processed_csv : str):
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self)-> str:
        df = pd.read_csv(self.original_csv, encodeing = "utf-8", error_bad_lines=False).dropna()
        required_cols = {"Name", "Genres", "sypnopsis"}
        
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError("There is a missing column in CSV File")
        
        df["combined_info"] = (
            "Title: " + df["Name"] + ", Overview: " + df["sypnopsis"] + ", Genres: " + df["Genres"]
        )

        df[["combined_info"]].to_csv(self.processed_csv, index=False, encodeing="utf-8")

        return self.processed_csv


