import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

curvy = df[abs(df['joy_curvature']) > 0.2].drop(columns=["Unnamed: 0"])

straight = df[abs(df['joy_curvature']) <= 0.2].drop(columns=["Unnamed: 0"])


curvy.to_csv("curv_greater_0.2.csv")
straight.to_csv("curv_smaller_0.2.csv")

