import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('C:/usr/home/iot2026/wine-analysis/date/wine.csv')

# 説明変数（自由に選べる）
X = df[['alcohol', 'volatile acidity', 'sulphates']].to_numpy()

# 切片項（1の列）を追加
X = np.hstack([np.ones((X.shape[0], 1)), X])

# 目的変数
y = df['quality'].to_numpy()

# 正規方程式で係数を求める (X^T X)^(-1) X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
beta = XtX_inv @ Xty

# 予測
y_pred = X @ beta

# 結果表示
print("回帰係数（切片 + 各説明変数）:", beta)

# 図の作成
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression (NumPy only)")
plt.grid(True)

# 図を保存（wine-analysis 直下）
plt.savefig('../linear_regression_result.png')

plt.show()
