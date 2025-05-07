# Analysis packages
import pandas as pd

# Custom functions for this analysis
import custom_functions as cf

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/target_data.csv')
com_data = pd.read_csv(f'{data_location}/combo_data.csv')

# Check how many nulls are in each feature
cf.print_nulls(tar_data)
cf.print_nulls(com_data)

# Plot feature distributions
# Only doing a subset of rows because otherwise it takes too long to plot
cf.plot_feature_distributions([
    com_data,
    tar_data,
    ],'raw_features')
cf.plot_MI(com_data[:len(tar_data)],tar_data,'raw_features')
cf.plot_correlations(com_data,'raw_features')
print('Saving plots to Plots')
