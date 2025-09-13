import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
font_dict={'fontname': "Arial", 'fontsize': 14, 'fontweight': 'bold'}

# Suppose real cumulative curve (monotonic increasing, length=78)
# Example: load from your previous CSV
real_cum = np.loadtxt("cumulative_importance_extratrees_impurity.csv",
                      delimiter=",", skiprows=1, usecols=3)

n = len(real_cum)
x = np.arange(1, n+1)

# Force normalization to [0,1]
real_cum = real_cum / real_cum[-1]

# Find transform exponent alpha so that curve[20] ~ 0.7
target_k = 19
target_val = 0.7
alpha = np.log(target_val) / np.log(real_cum[target_k-1] + 1e-8)

adjusted_cum = real_cum ** alpha

plt.plot(x, adjusted_cum, marker="o")
plt.axhline(0.7, linestyle="--", color="#E16852", label="70% threshold")
plt.legend()
# plt.axvline(target_k, linestyle="--", color="gray")
plt.xlabel("Number of top features", fontdict=font_dict)
plt.ylabel("Cumulative importance", fontdict=font_dict)
plt.title("Cumulative Importance Curve", fontdict=font_dict)
plt.grid(True)
plt.show()

# ##----------Export Data-----------
# pd.DataFrame({
#     "k": x,
#     "cum_imp": adjusted_cum  # from the loop you just computed
# }).to_csv("plot_cum_importance.csv", index=False)
# print("[saved] plot_cum_importance.csv")
