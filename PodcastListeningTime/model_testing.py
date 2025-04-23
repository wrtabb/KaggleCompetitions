import time
import threading
import psutil
import matplotlib.pyplot as plt

# Analysis packages
import pandas as pd

# Packages for feature engineering
from sklearn.impute import SimpleImputer

# Custom functions for this analysis
import podcast_functions as pf

# Model building packages
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV

mem_usage = []
timestamps = []
start_time = time.time()

def track_memory(interval=1):
    """Log RAM usage at regular intervals."""
    while True:
        mem_usage.append(psutil.virtual_memory().used / (1024 ** 3))  # Convert to GB
        timestamps.append(time.time() - start_time)
        time.sleep(interval)

tracking_thread = threading.Thread(target=track_memory, daemon=True)
tracking_thread.start()

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

print(f'Features to train on: \n{X.columns}')
print(f'Number of features: {len(X.columns)}')
print(type(X))
print(type(X_test))
print(type(y))

# Here I am using LazyRegressor to quickly evaluate how well a bunch of different models perform on this data
test_rows = 20000
X_trunc = X[:test_rows]
y_trunc = y[:test_rows]
x_train_lazy , x_test_lazy , y_train_lazy , y_test_lazy = train_test_split(X_trunc, y_trunc, test_size=0.2, random_state=0, shuffle=True)
lazy_model = LazyRegressor(verbose=1, random_state=0, regressors='all')
train_lazy , test_lazy = lazy_model.fit(x_train_lazy, x_test_lazy, y_train_lazy, y_test_lazy)

test_lazy.to_csv('Data/lazy_test_results.csv')
print(test_lazy)
print(train_lazy)

plt.figure(figsize=(10, 5))
plt.axhline(y=8, color='r', linestyle='--', label="max ram")
plt.plot(timestamps, mem_usage, label="RAM Usage (GB)")
plt.xlabel("Time (seconds)")
plt.ylabel("RAM Used (GB)")
plt.title("RAM Usage Over Time")
plt.legend()
plt.show()
plt.savefig("Plots/ram_usage.png", dpi=300, bbox_inches='tight')
