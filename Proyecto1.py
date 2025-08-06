import pandas as pd
import numpy as np

pro = pd.read_csv("proyecto1.csv", na_values=[" ", ""])

print(pro)
print(pro.describe())   

print("_"*30)
print(pro.isnull().sum())

pro.dropna(inplace=True)
print(pro)