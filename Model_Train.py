import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_excel("D:\coffee-revenue-predictor\coffee_shop_sales_dataset.xlsx")

df['Date'] = pd.to_datetime(df['Date'])
# Create additional time-based features
df['Day_of_Year'] = df['Date'].dt.dayofyear
df['Week_of_Year'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter

le = LabelEncoder()
df['Day_Name_Encoded'] = le.fit_transform(df['Day_Name'])
df['Season_Encoded'] = le.fit_transform(df['Season'])

# Select features for modeling (exclude target and non-predictive columns)
exclude_cols = ['Date', 'Day_Name', 'Season', 'Daily_Revenue', 'Staff_Cost',
                'Ingredient_Cost', 'Utilities_Cost', 'Rent_Cost', 'Total_Costs',
                'Daily_Profit']

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['Daily_Revenue']

print(f"✓ Features selected: {len(feature_cols)} columns")
print(f"✓ Target variable: Daily_Revenue")

# Basic statistics
print("\nTarget Variable (Daily_Revenue) Statistics:")
print(f"Mean: ${y.mean():.2f}")
print(f"Median: ${y.median():.2f}")
print(f"Std Dev: ${y.std():.2f}")
print(f"Min: ${y.min():.2f}")
print(f"Max: ${y.max():.2f}")

print("\n🔗 Top 10 Features Correlated with Daily Revenue:")
correlations = df[feature_cols + ['Daily_Revenue']].corr()['Daily_Revenue'].sort_values(ascending=False)
print(correlations.head(11)[1:])

print("\n Now i find which one is more important feature to my Model : Coffee_Sales")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaled using the Standard Scaler")
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print(f"\n🎯 Feature Selection: Top {len(selected_features)} features selected")
print("Selected features:", list(selected_features))

print("Training the Linear Regression Model")
model = LinearRegression()
model.fit(X_train_selected, y_train)
print("Model Train Successfully")

y_pred_train = model.predict(X_train_selected)
y_pred_test = model.predict(X_test_selected)


# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n📊 {dataset_name} Set Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


# Evaluate model performance
train_metrics = calculate_metrics(y_train, y_pred_train, "Training")
test_metrics = calculate_metrics(y_test, y_pred_test, "Test")

# Check for overfitting
print(f"\n🔍 Overfitting Check:")
print(f"R² difference (Train - Test): {train_metrics['R2'] - test_metrics['R2']:.4f}")
if abs(train_metrics['R2'] - test_metrics['R2']) < 0.05:
    print("✓ Model appears to generalize well (low overfitting)")
else:
    print("⚠️ Potential overfitting detected")

import joblib

# save the model
joblib.dump(model, "coffee_sales_model.pkl")
print("✓ Model saved as coffee_sales_model.pkl")

joblib.dump(scaler, "scaler.pkl")
print("✓ Scaler saved as scaler.pkl")

joblib.dump(selector, "selector.pkl")
print("✓ Selector saved as selector.pkl")
