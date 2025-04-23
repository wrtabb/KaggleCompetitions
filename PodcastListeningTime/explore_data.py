# Analysis packages
import pandas as pd

# Packages for feature engineering
from sklearn.impute import SimpleImputer

# Custom functions for this analysis
import podcast_functions as pf

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/tar_data.csv')
cat_data = pd.read_csv(f'{data_location}/cat_data.csv')
num_data = pd.read_csv(f'{data_location}/num_data.csv')

# Check how many nulls are in each feature
pf.print_nulls(tar_data)
pf.print_nulls(num_data)
pf.print_nulls(cat_data)

# Before dealing with nulls, add feature for whether a given feature has a NaN or not
col_with_nans = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in col_with_nans:
    new_col = f'{col}_NaN'
    num_data[new_col] = num_data[col].isna().astype(int)

imputer_num = SimpleImputer(strategy='median')
num_data = pd.DataFrame(imputer_num.fit_transform(num_data),columns=num_data.columns)
pf.print_nulls(num_data)

# Plot feature distributions
# Only doing a subset of rows because otherwise it takes too long to plot
numrows_to_plot = 10000
pf.plot_feature_distributions([
    num_data[:numrows_to_plot],
    cat_data[:numrows_to_plot],
    tar_data[:numrows_to_plot]
    ])

tar_data.to_csv(f'{data_location}/tar_data.csv', index=False)
cat_data.to_csv(f'{data_location}/cat_data.csv', index=False)
num_data.to_csv(f'{data_location}/num_data.csv', index=False)

