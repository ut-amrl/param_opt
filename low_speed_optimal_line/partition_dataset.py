import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

curvy = df[abs(df['joy_curvature']) > 0.2]

straight = df[abs(df['joy_curvature']) <= 0.2]

straight.to_csv("curv_greater_0.2.csv")
curvy.to_csv("curv_smaller_0.2.csv")

