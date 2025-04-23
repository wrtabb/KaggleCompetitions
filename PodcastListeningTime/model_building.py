import podcast_functions as pf
import pandas as pd

# Load data
data_location = 'Data'
tar_data = pd.read_csv(f'{data_location}/tar_data.csv')
combo_data = pd.read_csv(f'{data_location}/combo_data.csv')

# Split testing and training data once again
index_split = len(tar_data)
y = tar_data.copy()
X = combo_data[:index_split].copy()
X_test = combo_data[index_split:].copy()

del combo_data
del tar_data

# Training each individual model
models = []
models.append(pf.train_HuberRegressor(X,y))
#models.append(pf.train_HistGradientBoostingRegressor(X,y))
#models.append(pf.train_GradientBoostingRegressor(X,y))
#models.append(pf.train_XGBoost(X,y))
#models.append(pf.train_LGBMRegressor(X,y))

# Train the stacking model containing all models
model = pf.train_StackingModel(X,y,models)
prediction = model.predict(X_test)

# Save the results applied to the test data
#pf.SaveOutputFilesForCompetition(X_test,prediction)


