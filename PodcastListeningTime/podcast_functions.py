import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import clear_output
from lazypredict.Supervised import LazyRegressor
from scipy.stats import uniform, randint
from lightgbm import LGBMRegressor

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

def train_HuberRegressor(X_train, y_train):
    print('Starting to train HuberRegressor\n')

    params = {
        'epsilon': 6.309137428783002,
        'alpha': 0.004877974938442685,
        'max_iter': 200,
        'fit_intercept': True,
        'tol': 0.0008494667285558747
    }

    huber = HuberRegressor(**params)
    huber.fit(X_train, y_train)
    return huber

def train_HistGradientBoostingRegressor(X_train, y_train):
    print('Starting to train HistGradientBoostingRegressor\n')

    params = {
        'max_iter': 427,
        'learning_rate': 0.09571615180145268,
        'max_depth': 11,
        'min_samples_leaf': 84,
        'l2_regularization': 0.7152624894796115,
        'max_bins': 206,
    }

    hgb = HistGradientBoostingRegressor(**params)
    hgb.fit(X_train,y_train)

    return hgb

def train_GradientBoostingRegressor(X_train,y_train):
    print('Starting to train GradientBoostingRegressor\n')
    params = {
        'n_estimators': 393,
        'learning_rate': 0.06693690822892108,
        'max_depth': 12,
        'min_samples_split': 7,
        'min_samples_leaf': 8,
        'subsample': 0.9414365090985378,
        'max_features': None
    }
    gbr = GradientBoostingRegressor(**params)
    gbr.fit(X_train,y_train)
    return gbr

def train_XGBoost(X_train,y_train):
    print('Starting to train XGBRegressor\n')
    params = {
        "n_estimators": 462,
        "learning_rate": 0.08154792163442738,
        "max_depth": 12,
        "subsample": 0.9058954745308208,
        "colsample_bytree": 0.9456526638491439,
        "random_state": 0,
        "gamma": 0.33164411963508833,
        "reg_alpha": 0.01428218246267099,
        "reg_lambda": 0.8988143668046589
    }
    # XGBoost model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train,y_train)
    return model

def train_LGBMRegressor(X_train, y_train):
    print('Starting to train LGBMRegressor\n')
    params = {
        'n_estimators': 457,
        'learning_rate': 0.014915560132499209,
        'max_depth': 3,
        'num_leaves': 72,
        'min_child_samples': 19,
        'subsample': 0.7381421172094151,
        'colsample_bytree': 0.9449478096939333,
        'reg_alpha': 0.685559421931073,
        'reg_lambda': 0.3105632580080139,
        'random_state': 42
    }
    lgbm = LGBMRegressor(**params)
    lgbm.fit(X_train,y_train)
    return lgbm

def train_StackingModel(X,y,model_list):
    num_models = len(model_list)
    print(f'Creating stacking model from {num_models} models\n')
    model_names = [f'model_{idx}' for idx in range(num_models)]
    estimators = list(zip(model_names, model_list))  # THIS is the fix
    model_names = []
    cv_fold = KFold(n_splits= 5,shuffle=True,random_state=42)
    model = StackingRegressor(
        estimators=estimators,
        cv=cv_fold
    )

    model.fit(X,y)
    return model

def SaveOutputFilesForCompetition(df,preds):
    print('Saving output files\n')
    idx_array = np.arange(750000,1000000,dtype=int)
    df_idx = pd.Series(idx_array)
    print(df_idx)
    output = pd.DataFrame({'id': df_idx,
                       target: preds})
    output.to_csv('Data/submission.csv', index=False)
