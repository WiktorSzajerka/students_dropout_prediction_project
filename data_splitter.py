import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_csv(r"data\data.csv")
targets = data.iloc[:, -1]
features = data.drop(data.columns[-1],axis=1)

skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=12)

for i, (train, test) in enumerate(skf.split(features, targets)):
    data.iloc[train].to_csv(f"data/splits/fold{i}train.csv")
    data.iloc[test].to_csv(f"data/splits/fold{i}test.csv")
    