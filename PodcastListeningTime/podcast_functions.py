import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

def split_target(df,target):
    '''
        Splits the target from the training data.
        df is the dataframe
        target is the name of the target column
    '''
    target_data = df[target].copy()
    df.drop(target,axis=1,inplace=True)
    return target_data

def check_for_duplicated_rows(df):
    '''
        Checks to see if any rows are duplicates.
        df is the dataframe to be checked
    '''
    duplicates = df.duplicated()
    duplicated_rows = duplicates[duplicates]
    num_duplicates = len(duplicated_rows)
    print(f'There are {num_duplicates} duplicated rows')
    if num_duplicates>0:
        print('Removing these now ...')
        df.drop(df[df.duplicated()].index,inplace=True)

def print_nulls(df):
    '''
        Prints the number of null values in each column
        df is the dataframe to be checked for null values
    '''
    print(f'\nNull values:\n{df.isnull().sum()}')

def plot_feature_distributions(df_list):
    '''
        Plots the distribution for each feature in the dataframe
        df_list is a list of dataframes to be plotted
    '''
    print('\nStarting plot_feature_distributions()')
    df = pd.concat(df_list,axis=1)
    num_features = len(df.columns)
    print(f'Number of features to plot = {num_features}')

    ncols = 3
    nrows = math.ceil(num_features/ncols)
    width = 10*ncols
    height = 10*nrows
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(width,height))
    for i, feature in enumerate(df.columns):
        sns.histplot(data=df, x=feature, ax=axes[i%nrows,i//nrows], color='purple')
    plt.savefig(f'Plots/feature_distributions.png', dpi=300, bbox_inches='tight')

def frequency_encoding(df,cat_cols):
    '''
        Carries out frequency encoding on categorical features
        df is a dataframe
        cat_cols is a list of of categorical feature names
    '''
    print(f'Freq encoding for {cat_cols}')
    new_names = [col+'_freq' for col in cat_cols]
    print(f'New column names: {new_names}')
    df[new_names] = df[cat_cols].apply(
            lambda col: col.map(col.value_counts(normalize=True)))

    df.drop(cat_cols,axis=1,inplace=True)
    return df

def plot_MI(df,tar,save_name):
    '''
        Finds the mutual information between features and the target
        to determine which features are most important for predicting the target
        Creates a plot and saves it
        df is dataframe
        tar is the name of the target
        save_name is the name you want the plot saved as
    '''

    print('\nStarting plot_MI()')
    df_target = tar.values.ravel()
    df_train = df
    mi = mutual_info_regression(df_train, df_target)
    features = df_train.columns
    mi_dict = dict(zip(features, mi))
    plt.figure(figsize=(35, 10),dpi=500)
    plt.bar(features, mi, color='skyblue')
    plt.title('Mutual Information Between Features and Target')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')
    plt.savefig(f'Plots/{save_name}.png', dpi=300, bbox_inches='tight')
    return mi_dict

def plot_correlations(df,save_name):
    '''
        Makes a plot of all feature correlations
        df is dataframe
        save_name is the name you want the plot saved as
    '''
    print('\nStarting plot_correlations()')
    df_all = df.copy()
    corr_data = df_all.corr(method='pearson')  # Get the correlation matrix

    plt.figure(figsize=(15,15))
    sns.heatmap(data=corr_data, cmap='coolwarm', annot=True, fmt='.2g')
    plt.savefig(f'Plots/{save_name}.png', dpi=300, bbox_inches='tight')
