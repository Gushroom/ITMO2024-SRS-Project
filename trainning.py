import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
data = pd.read_csv("velocity.csv")

# Assume the data is in chronological order
# Split into features and targets
X = data[['omega_left', 'omega_right']]
y = data[['linear', 'angular']]

# Split the data sequentially
train_size = int(0.8 * len(data))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Feature scaling for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# MLP Model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Evaluation Metrics for XGBoost
mse_v_xgb = mean_squared_error(y_test['linear'], y_pred_xgb[:, 0])
mae_v_xgb = mean_absolute_error(y_test['linear'], y_pred_xgb[:, 0])
mse_w_xgb = mean_squared_error(y_test['angular'], y_pred_xgb[:, 1])
mae_w_xgb = mean_absolute_error(y_test['angular'], y_pred_xgb[:, 1])

# Evaluation Metrics for MLP
mse_v_mlp = mean_squared_error(y_test['linear'], y_pred_mlp[:, 0])
mae_v_mlp = mean_absolute_error(y_test['linear'], y_pred_mlp[:, 0])
mse_w_mlp = mean_squared_error(y_test['angular'], y_pred_mlp[:, 1])
mae_w_mlp = mean_absolute_error(y_test['angular'], y_pred_mlp[:, 1])

# Print the metrics
print("XGBoost Model Performance:")
print(f"MSE for v: {mse_v_xgb}, MAE for v: {mae_v_xgb}")
print(f"MSE for w: {mse_w_xgb}, MAE for w: {mae_w_xgb}")
print("\nMLP Model Performance:")
print(f"MSE for v: {mse_v_mlp}, MAE for v: {mae_v_mlp}")
print(f"MSE for w: {mse_w_mlp}, MAE for w: {mae_w_mlp}")

# Save XGBoost model
xgb_model.save_model('xgboost_model.json')