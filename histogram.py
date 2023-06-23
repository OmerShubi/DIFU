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


round_units = 3
taus = []
rs = []
rhos = []
from scipy.stats import kendalltau, pearsonr, spearmanr
for annotator1, annotator2 in zip(['Shubi','Shubi','Tamir'], ['Itay','Tamir','Itay']):
    annot1 = df2[annotator1]
    Annot2 = df2[annotator2]
    tau, _ = kendalltau(annot1, Annot2)

    # Compute Pearson correlation
    r, _ = pearsonr(annot1, Annot2)

    # Compute Spearman correlation
    rho, _ = spearmanr(annot1, Annot2)
    # print annotator names and the three correlations in one print statement
    print(f'{annotator1} and {annotator2} have Kendall\'s tau of {round(tau,round_units)}, Pearson correlation of {round(r,round_units)}, and Spearman correlation of {round(rho,round_units)}.')
    taus.append(tau)
    rs.append(r)
    rhos.append(rho)
# %%
print(f'Average Kendall\'s tau is {round(sum(taus)/len(taus),round_units)}')
print(f'Average Pearson correlation is {round(sum(rs)/len(rs),round_units)}')
print(f'Average Spearman correlation is {round(sum(rhos)/len(rhos),round_units)}')
# %%


#%%
df = pd.read_csv('part3A_team3_agreement.csv')

df[['base_example_id', 'type', 'topic']] = df['example_id'].str.split('_', expand=True)

df = df[['type','topic','agreement']].dropna()

# split label column into two columns based on the first character
df['numeric_label'] = df['agreement'].astype(int)
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
# %%
# %%
sns.histplot(df[df['type']=='1'], x='numeric_label', 
              discrete=True,
             hue='topic', multiple='stack')
plt.xlabel("Guilt")
# stacked histplot

# seaborn hide legend title
plt.legend(title='Topic', labels=['Relationship', 'Work','Family'])
plt.xticks([1,2,3,4])
#plt.xlabel('Corrected score')
plt.show()
# %%
