import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
font_dict={'fontname': "Arial", 'fontsize': 14, 'fontweight': 'bold'}

df = pd.read_csv('tradeoff_adjust.csv')

k = df['k']
mean_scores = df['cv_accuracy']

plt.figure(figsize=(7,5))
plt.plot(range(5,79,5), mean_scores, marker="o")
plt.xlabel("Number of top features", fontdict=font_dict)
plt.ylabel("Accuracy", fontdict=font_dict)
plt.title("Performance vs. Feature Count Trade-off", fontdict=font_dict)
plt.grid(True)
plt.show()
