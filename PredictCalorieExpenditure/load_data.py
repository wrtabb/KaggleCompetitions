# Analysis packages
import pandas as pd

# Custom functions for this analysis
import custom_functions as cf

data_loc = 'Data'

# Load training and testing data
train_data = pd.read_csv(f'{data_loc}/train.csv')
test_data = pd.read_csv (f'{data_loc}/test.csv')
target = 'Calories'

# compare columns in the dataframe
print(f'train_data columns: \n{train_data.columns}')

# same columns except for id, which will be dropped here
train_data.drop('id',axis=1,inplace=True)
test_data.drop('id',axis=1,inplace=True)

# Split target from the training data using custom function
target_data = cf.split_target(train_data,target)

# combine training and testing data for data exploration and feature engineering
combo_data = pd.concat([train_data,test_data])

# There is only one categorical feature and it is binary
# I'll encode that here as an integer
combo_data['male'] = (combo_data['Sex']=='male').astype(int)
combo_data.drop('Sex',axis=1,inplace=True)
print(combo_data.head())

print(f'Training data shape: {train_data.shape}')
print(f'Testing data shape: {test_data.shape}')
print(f'Combination data shape: {combo_data.shape}')
print(f'Target data shape: {target_data.shape}')

# Save all data to new csv files
print(f'Saving all data to {data_loc}/*.csv')
target_data.to_csv(f'{data_loc}/target_data.csv',index=False)
combo_data.to_csv(f'{data_loc}/combo_data.csv',index=False)
