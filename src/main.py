import pandas as pd
import os

csv_path = os.path.join("resources", "STAXI.csv")
staxi = pd.read_csv(csv_path, encoding="utf-8")
print(staxi.head())

csv_path2 = os.path.join("resources", "TAS.csv")
tas = pd.read_csv(csv_path2, encoding="utf-8")
print(tas.head())
