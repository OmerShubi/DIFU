#%%
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_correlation(df, annotator1_name, annotator2_name, round_units=3):

    df1 = df[f"{annotator1_name}_numeric_label"]
    df2 = df[f"{annotator2_name}_numeric_label"]
    # Compute Kendall's tau
    tau, _ = kendalltau(df1, df2)

    # Compute Pearson correlation
    r, _ = pearsonr(df1, df2)

    # Compute Spearman correlation
    rho, _ = spearmanr(df1, df2)
    # print annotator names and the three correlations in one print statement
    print(f'{annotator1_name} and {annotator2_name} have Kendall\'s tau of {round(tau,round_units)}, Pearson correlation of {round(r,round_units)}, and Spearman correlation of {round(rho,round_units)}.')

def compute_confusion(df, annotator1_name, annotator2_name, data_type, annonimuous=True):
    df1 = df[f"{annotator1_name}_numeric_label"]
    df2 = df[f"{annotator2_name}_numeric_label"]
    # Compute confusion matrix
    cm = confusion_matrix(y_true=df1, y_pred=df2)
    # number unique labels
    num_labels = df1.nunique()
    # Define labels for the confusion matrix
    if num_labels == 5:
        labels = ['Not Guilty', 'Slightly Guilty', 'Moderately Guilty', 'Very Guilty', 'Completely Guilty']
    elif num_labels == 4:
        labels = ['Not Guilty', 'Slightly Guilty', 'Very Guilty', 'Completely Guilty']
        
    # Plot confusion matrix as heatmap
    s = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, 
    xticklabels=labels, yticklabels=labels)
    # Add x axis label
    s.set_xlabel(annotator1_name if not annonimuous else 'Annotator 1')
    # Add y axis label
    s.set_ylabel(annotator2_name if not annonimuous else 'Annotator 2')
    s.set_title(f'Confusion Matrix - {data_type}')
    plt.tight_layout()
    # Show plot
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    
def load_and_process(path1, path2, annotator1_name, annotator2_name, split_id):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    # Concatenate columns
    df = pd.concat([df1, df2], keys=[annotator1_name, annotator2_name], axis=1)
    df = df.dropna()
    # split label column into two columns based on the first character
    for annotator_name in [annotator1_name, annotator2_name]:
        try:
            df[(annotator_name,'numeric_label')] = df[(annotator_name,'label')].str[0].astype(int)
        except AttributeError:
            df[(annotator_name,'numeric_label')] = df[(annotator_name,'label')].astype(int)

    df.columns = ['_'.join(col) for col in df.columns.values]
    df.rename(columns={f'{annotator1_name}_example_id': 'example_id'}, inplace=True)
    if split_id:
        df[['base_example_id', 'type', 'topic']] = df['example_id'].str.split('_', expand=True)
    return df

#%%
if __name__ == "__main__":
    ANNOT1 = 'Shubi'
    ANNOT2 = 'Tamir'
    sht_full = load_and_process(path1=f'part2_team3_member_{ANNOT1} - Sheet1.csv', 
                                path2=f'part2_team3_member_{ANNOT2} - Sheet1.csv',
                                annotator1_name=ANNOT1, annotator2_name=ANNOT2, split_id=False)
    
    sht_long = sht_full.iloc[:100]
    sht_short = sht_full.iloc[100:]
    for data_type, data_type_name in zip([sht_full, sht_long, sht_short], ['Full story + TL;DR', 'Full story', 'TL;DR']):
        print(data_type_name)

        compute_correlation(data_type, annotator1_name=ANNOT1, annotator2_name=ANNOT2)
        compute_confusion(data_type,  annotator1_name=ANNOT1, annotator2_name=ANNOT2, data_type=data_type_name)
    
    #%%
    ANNOT1 = 'sahar'
    ANNOT2 = 'tomer'
    st_all = load_and_process(path1=f'part3A_team3_{ANNOT1}.csv', 
                                path2=f'part3A_team3_{ANNOT2}.csv',  
                                annotator1_name=ANNOT1, annotator2_name=ANNOT2, split_id=True)
    
    st_long = st_all[st_all.type=='2']
    st_short = st_all[st_all.type=='1']
    
    for data_type, data_type_name in zip([st_all, st_long, st_short], ['Full story + TL;DR', 'Full story', 'TL;DR']):
        print(data_type_name)
        compute_correlation(data_type,  annotator1_name=ANNOT1, annotator2_name=ANNOT2)
        compute_confusion(data_type,  annonimuous=True, annotator1_name=ANNOT1,  annotator2_name=ANNOT2, data_type=data_type_name)
        
 

# %%
