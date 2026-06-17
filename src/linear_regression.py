import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/usr/home/iot2026/wine-analysis/date/wine.csv')

X = df[['alcohol']].to_numpy()
X = np.hstack([np.ones((X.shape[0], 1)), X])
y = df['quality'].to_numpy()

XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
beta = XtX_inv @ Xty

y_pred = X @ beta

print("回帰係数（切片 + alcohol）:", beta)

plt.figure(figsize=(8, 5))
plt.scatter(df['alcohol'], y, alpha=0.5)
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Linear Regression: Alcohol vs Quality')
plt.grid(True)

plt.savefig('../linear_regression_result.png')
plt.show()
