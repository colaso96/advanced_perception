# Bronte Sihan Li, Cole Crescas Dec 2023
# CS7180

import xgboost as xgb
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

# Load the large CSV file
df = pd.read_csv('/notebooks/rapids/xgboost/train_added_features.csv')
df = df.fillna(df.mean())

# Assuming the target variable is in the 'target' column
target_column = 'target'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DMatrix for training and testing
dtrain = xgb.DMatrix(X_train, label=y_train)
# Specify the parameter grid for RandomizedSearchCV
param_dist = {
    'objective': ['reg:squarederror'],
    'eval_metric': ['mae'],
    'max_depth': [3, 6, 9],
    'eta': [0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'tree_method': ["gpu_hist"],  # Use GPU acceleration
    #'device': ['gpu'],
}

# Create an XGBoost regressor
xgb_reg = xgb.XGBRegressor()

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    xgb_reg,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Train the XGBoost model with the best hyperparameters
best_model = xgb.train(best_params, dtrain, 100)

# Make predictions on the test set
dtest = xgb.DMatrix(X_test)
y_pred = best_model.predict(dtest)

# Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error with Hyperparameter Tuning with XGBoost: {mae}')

# ------------------------------ Light GBM model -------------------------------------------

# LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters for LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}

# Train the LightGBM model
num_round = 1000  # You can adjust this based on your dataset
bst = lgb.train(lgb_params, train_data, num_round, valid_sets=[val_data])

# Make predictions on the validation set
y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

# Calculate MAE on the validation set
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error on Validation Set with LGBM: {mae}')
