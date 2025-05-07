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

def print_nulls(df):
    '''
        Prints the number of null values in each column
        df is the dataframe to be checked for null values
    '''
    print(f'\nNull values:\n{df.isnull().sum()}')

def plot_feature_distributions(df_list,save_tag):
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
    plt.savefig(f'Plots/feature_distributions_{save_tag}.png', dpi=300, bbox_inches='tight')

def plot_MI(df,tar,save_tag):
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
    plt.savefig(f'Plots/mutual_information_{save_tag}.png', bbox_inches='tight')
    return mi_dict

def plot_correlations(df,save_tag):
    '''
        Makes a plot of all feature correlations
        df is dataframe
        save_name is the name you want the plot saved as
    '''
    print('\nStarting plot_correlations()\nThis may take a while to run')
    df_all = df.copy()
    corr_data = df_all.corr(method='pearson')  # Get the correlation matrix

    plt.figure(figsize=(15,15))
    sns.heatmap(data=corr_data, cmap='coolwarm', annot=True, fmt='.2g')
    plt.savefig(f'Plots/correlations_{save_tag}.png', dpi=300, bbox_inches='tight')

def train_XGBoost(X_train,y_train,params):
    print('Starting to train XGBRegressor\n')
    # XGBoost model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train,y_train)
    return model

def SaveOutputFilesForCompetition(df,preds):
    print('Saving output files\n')
    idx_array = np.arange(750000,1000000,dtype=int)
    df_idx = pd.Series(idx_array)
    print(df_idx)
    output = pd.DataFrame({'id': df_idx,
                       target: preds})
    output.to_csv('Data/submission.csv', index=False)

def create_features(df):
    # Use BMI to utilize height and weight information together
    df['BMI'] = df['Weight']/df['Height']**2

    # difference between body temperature and the average body temperature
    df['temp_diff'] = df['Body_Temp']-37.0

    # The total number of heartbeats may be correlated with the target
    df['heartbeats'] = df['Heart_Rate']*df['Duration']

    return df

def get_parameters_XGBoost(X_train,y_train):
    print('Finding best parameters for XGBoost')
    param_dist = {
        'n_estimators': randint(100,500),
        'learning_rate': uniform(0.01,0.1),
        'max_depth': randint(4,16),
        'subsample': uniform(0.7,0.3),
        'colsample_bytree': uniform(0.7,0.3),  # instead of max_features
        'gamma': uniform(0,5),
        'reg_alpha': uniform(0,1),
        'reg_lambda': uniform(0,1)
    }
    model = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    random_search.fit(X_train,y_train)
    print("Best XGB Parameters:", random_search.best_params_)
    return random_search.best_params_

def rmsle(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    print(f'y_true has {len(y_true)} entries\ny_pred has {len(y_pred)} entries')
    print(f'Min of y_true: {y_true.min()}, \nMin of y_pred: {y_pred.min()}')

    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, prediction)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

