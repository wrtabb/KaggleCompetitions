import custom_functions as cf
import pandas as pd

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/tar_data.csv')
com_data = pd.read_csv(f'{data_location}/combo_data.csv')

# Split testing and training data  
y = tar_data.copy()
X = com_data[:len(tar_data)].copy()
X_test = com_data[len(tar_data):].copy()

print(f'Features to train on: \n{X.columns}')
print(f'Number of features: {len(X.columns)}')
print(type(X))
print(type(X_test))
print(type(y))

del com_data
del tar_data

# Determine best parameters from a custom function
best_params = cf.get_parameters_XGBoost(X,y)

# Create XGBoost model
model_xgb = cf.train_XGBoost(X,y,best_params)
prediction = model_xgb.predict(X)
print(f'RMSLE: {cf.rmsle(y,prediction)}')

# Save the results applied to the test data
#pf.SaveOutputFilesForCompetition(X_test,prediction)


