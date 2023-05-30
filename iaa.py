#%%
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
import pandas as pd
df1 = pd.read_csv('/Users/shubi/DIFU/part2_team3_member_Shubi - Sheet1.csv')
df2 = pd.read_csv('/Users/shubi/DIFU/part2_team3_member_Tamir - Sheet1.csv')


# Concatenate columns
df = pd.concat([df1, df2], keys=['shubi', 'tamir'], axis=1)

df = df.dropna()

# split label column into two columns based on the first character
df[('shubi','numeric_label')] = df[('shubi','label')].str[0].astype(int)
df[('tamir','numeric_label')] = df[('tamir','label')].str[0].astype(int)

#%%
# Example data
shubi = df[('shubi','numeric_label')]
tamir = df[('tamir','numeric_label')]

# Compute Kendall's tau
tau, _ = kendalltau(shubi, tamir)
print("Kendall's tau:", round(tau,3))

# Compute Pearson correlation
r, _ = pearsonr(shubi, tamir)
print("Pearson correlation:", round(r,3))

# Compute Spearman correlation
rho, _ = spearmanr(shubi, tamir)
print("Spearman correlation:", round(rho,3))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_true=shubi, y_pred=tamir)

# Define labels for the confusion matrix
labels = ['Not Guilty', 'Slightly Guilty', 'Moderately Guilty', 'Very Guilty', 'Completely Guilty']

# Plot confusion matrix as heatmap
s = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, 
xticklabels=labels, yticklabels=labels)
# Add x axis label
s.set_xlabel('Tamir')
# Add y axis label
s.set_ylabel('Shubi')
plt.tight_layout()
# Show plot
plt.show()