#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid", {'axes.grid' : False})
df = pd.read_csv('data_team3 - DIFU.csv')
#%%
df = df[['type','topic','label']].dropna()
#%%
# split label column into two columns based on the first character
df['numeric_label'] = df['label'].str[0].astype(int)
#%%
# set figsize
plt.figure(figsize=(8, 6))
sns.histplot(df, x='numeric_label', discrete=False, common_norm=False,
             hue='type', multiple='dodge', stat='proportion')
plt.xlabel("Guilt")

# no space between bars

# seaborn hide legend title
#plt.legend(title='English L1', loc='upper left', labels=['Yes', 'No'])
#plt.xlabel('Corrected score')
plt.show()

#%%
df['topic'].value_counts()
# %%
sns.histplot(df[df['type']=='documents'], x='numeric_label',
              discrete=True,
             hue='topic', multiple='stack')
plt.xlabel("Guilt")
# stacked histplot

# seaborn hide legend title
#plt.legend(title='English L1', loc='upper left', labels=['Yes', 'No'])
#plt.xlabel('Corrected score')
plt.show()
# %%
df2 = pd.read_csv('data_team3 - DIFU.csv')
df2
# %%
df2=df2[['Shubi','Itay','Tamir']].dropna()

# %%
df2

# %%
# drop rows that cannot be converted to float
df2 = df2.apply(pd.to_numeric, errors='coerce').dropna()
df2
# %%
# compute inter-rater reliability
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(df2['Shubi'], df2['Itay'])
#%%
cohen_kappa_score(df2['Shubi'], df2['Tamir'])


# %%
cohen_kappa_score(df2['Tamir'], df2['Itay'])

# %%
