# Analysis packages
import pandas as pd

# Custom functions for this analysis
import podcast_functions as pf

# Load training and testing data
data_location = 'Data'
train_data = pd.read_csv(f'{data_location}/train.csv')
train_data_ext = pd.read_csv(f'{data_location}/podcast_dataset.csv')
test_data = pd.read_csv(f'{data_location}/test.csv')
target = 'Listening_Time_minutes'

# compare columns in the dataframe
print(f'train_data columns: \n{train_data.columns}')
print(f'train_data_ext columns: \n{train_data_ext.columns}')

# same columns except for id, which will be dropped here
train_data.drop('id',axis=1,inplace=True)
test_data.drop('id',axis=1,inplace=True)

# Check for duplicated rows
pf.check_for_duplicated_rows(train_data)
pf.check_for_duplicated_rows(train_data_ext)
pf.check_for_duplicated_rows(test_data)

# There are duplicates in train_data_ext, so make sure they were removed by the function
pf.check_for_duplicated_rows(train_data_ext)

# Combine training data with the original dataset and split the target from it
train_data = pd.concat([train_data,train_data_ext])

# Remove rows for which the target doesn't exist before splitting target from training data
train_data = train_data.dropna(subset=[target])
tar_data = pf.split_target(train_data,target)

# combine training and testing data for ease of data exploration and feature engineering
combo_data = pd.concat([train_data,test_data])
index_split = int(len(train_data))
print(f'Index split = {index_split}')

# Separate categorical from numerical values
cat_cols = [col for col in combo_data.columns if 
            combo_data[col].dtype=='object'] 
num_cols = [col for col in combo_data.columns if 
           combo_data[col].dtype in ['float','int']]
all_cols = cat_cols+num_cols

# Will need to separately work on categorical and numerical data, so split into separate dataframes
cat_data = combo_data[cat_cols].copy()
num_data = combo_data[num_cols].copy()

print(f'Starting features for training data: \n{train_data.columns}')
print(f'Training data shape: {train_data.shape}')
print(f'Testing data shape: {test_data.shape}')
print(f'Combination data shape: {combo_data.shape}')
print(f'Categorical data shape: {cat_data.shape}')
print(f'Numerical data shape: {num_data.shape}')

tar_data.to_csv(f'{data_location}/tar_data.csv', index=False)
cat_data.to_csv(f'{data_location}/cat_data.csv', index=False)
num_data.to_csv(f'{data_location}/num_data.csv', index=False)
