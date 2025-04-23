# Analysis packages
import pandas as pd

# Packages for feature engineering
from sklearn.preprocessing import RobustScaler

# Custom functions for this analysis
import podcast_functions as pf

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/tar_data.csv')
cat_data = pd.read_csv(f'{data_location}/cat_data.csv')
num_data = pd.read_csv(f'{data_location}/num_data.csv')

# There are too many categories in some categorical variables
# Use frequency encoding
cat_data = pf.frequency_encoding(cat_data,cat_data.columns)

# Recombine categorical and numerical features into one dataframe
combo_data = pd.concat([cat_data.reset_index(drop=True), num_data.reset_index(drop=True)], axis=1)
del cat_data
del num_data

all_features = combo_data.columns
# Robust scaling
robust_data = RobustScaler()
robust_data.fit(combo_data)
combo_data = robust_data.transform(combo_data)

# RobustScalar turns the dataframe into a numpy array
# Turn it back into a pandas dataframe
combo_data = pd.DataFrame(combo_data,columns=all_features)

index_split = len(tar_data)
mi_dict = pf.plot_MI(combo_data[:index_split],tar_data,'mutual_information')
pf.plot_correlations(combo_data,'feature_correlations')

tar_data.to_csv(f'{data_location}/tar_data.csv', index=False)
combo_data.to_csv(f'{data_location}/combo_data.csv', index=False)
