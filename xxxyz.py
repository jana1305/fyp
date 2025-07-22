# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# --- Load Dataset ---
file_path = rbluebike_data (1).csv
df = pd.read_csv(file_path)

# --- Initial Cleaning ---
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns if any

# --- Summary Info ---
print(df.info())
print(df.describe())

# --- Check Missing Values ---
print(df.isnull().sum())

# --- Convert and Extract Temporal Features ---
df['Time Index (Hourly)'] = pd.to_datetime(df['Time Index (Hourly)'], dayfirst=True)
df['hour'] = df['Time Index (Hourly)'].dt.hour
df['month'] = df['Time Index (Hourly)'].dt.month
df['day'] = df['Time Index (Hourly)'].dt.day
df['year'] = df['Time Index (Hourly)'].dt.year

# --- Visualize Missing Values ---
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# --- Outlier Detection via Boxplot ---
num_cols = ['casual_riders_count', 'member_riders_count', 
            'casual_rider_duration', 'member_rider_duration', 
            'travel_time', 'Temp(c)', 'rel_humidity', 'wspd', 'pres']
plt.figure(figsize=(15, 10))
df[num_cols].boxplot()
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# --- Standardization ---
scaler_std = StandardScaler()
df_standardized = df.copy()
df_standardized[num_cols] = scaler_std.fit_transform(df[num_cols])

plt.figure(figsize=(15, 10))
df_standardized[num_cols].boxplot()
plt.title("Standardized Numerical Features")
plt.xticks(rotation=45)
plt.show()

# --- Normalization ---
scaler_norm = MinMaxScaler()
df_normalized = df.copy()
df_normalized[num_cols] = scaler_norm.fit_transform(df[num_cols])

# --- Histograms (Original vs Normalized) ---
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 12))
axs = axs.flatten()
for i, col in enumerate(num_cols):
    axs[i].hist(df[col], bins=30, alpha=0.6, label='Original')
    axs[i].hist(df_normalized[col], bins=30, alpha=0.6, label='Normalized')
    axs[i].set_title(col)
    axs[i].legend()
plt.tight_layout()
plt.show()

# --- Correlation Matrix ---
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Plot heatmap of correlation matrix
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# --- Feature Selection ---
features = ['Temp(c)', 'rel_humidity', 'wspd', 'pres', 'IsWeekend', 'IsHoliday', 'travel_time']
target = 'count'
X = df_normalized[features]
y = df_normalized[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 📌 Model 1: Linear Regression
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - MSE: {mse_lr:.4f}, R² Score: {r2_lr:.4f}")

# ---------------------------
# 📌 Model 2: Decision Tree
# ---------------------------
dt_model = DecisionTreeRegressor(random_state=42, max_depth=6)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree - MSE: {mse_dt:.4f}, R² Score: {r2_dt:.4f}")

# ---------------------------
# 📌 Model 3: Random Forest
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - MSE: {mse_rf:.4f}, R² Score: {r2_rf:.4f}")

# ---------------------------
# 📌 Model 4: Gradient Boosting
# ---------------------------
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting - MSE: {mse_gb:.4f}, R² Score: {r2_gb:.4f}")

# --- Save the Best Model ---
joblib.dump(gb_model, "gradient_boosting_model.pkl")
print("✅ Gradient Boosting model saved as 'gradient_boosting_model.pkl'")
