import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("banknote.txt", sep=",", header=None)

no_of_features = len(data.columns)
X = data.iloc[:, 0:no_of_features - 1]
y = data.iloc[:, no_of_features - 1]

X = X.to_numpy()
for b in np.arange(-1 * (0.01), 0.001):
    print(b)

opt_dict = {'1': 1, '2': 2}
norms = sorted([n for n in opt_dict])
print(norms)
