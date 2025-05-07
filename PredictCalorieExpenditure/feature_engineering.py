# Analysis packages
import pandas as pd

# Custom functions for this analysis
import custom_functions as cf

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/tar_data.csv')
com_data = pd.read_csv(f'{data_location}/combo_data.csv')

# Create new features
com_data = cf.create_features(com_data)
print(com_data.head())

cf.plot_feature_distributions([com_data,tar_data],'added_features')
cf.plot_MI(com_data[:len(tar_data)],tar_data,'added_features')
cf.plot_correlations(com_data,'added_features')

com_data.to_csv(f'{data_loc}/combo_data.csv',index=False)
