# ==============================
# 🏠 House Price Prediction - Preprocessing & Modeling
# ==============================

# ============================================================
# Step 1: Import library
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import skew


# ============================================================
# Step 2: --- Load datasets ---
# ============================================================
train = pd.read_csv("/home/hardik/Desktop/python_intern/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/home/hardik/Desktop/python_intern/house-prices-advanced-regression-techniques/test.csv")

print("✅ Data loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# --- Explore data ---
print("\n📊 Train Data Info:")
print(train.info())

print("\n🧾 Summary Statistics:")
print(train.describe())

# --- Check for missing values ---
print("\n🔍 Missing Values in Train Data:")
print(train.isnull().sum().sort_values(ascending=False).head(10))

# --- Visualize Missing Data ---
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap - Train Data")
plt.show()


# ============================================================
# Step 3: Log-transform target variable (SalePrice)
# ============================================================
train["SalePrice"] = np.log1p(train["SalePrice"])  # log(1 + y)
y = train["SalePrice"]

# Combine train and test for consistent preprocessing
all_data = pd.concat((train.drop(["SalePrice"], axis=1), test)).reset_index(drop=True)
print("Combined data shape:", all_data.shape)


# ============================================================
# Step 4: Handle Missing Values
# ============================================================

# Fill NA with 'None' for categorical columns that mean "No feature"
for col in [
    'PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType',
    'GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond',
    'BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType'
]:
    all_data[col] = all_data[col].fillna('None')

# Fill 0 for numerical features with no basement/garage/etc.
for col in [
    'GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
    'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea'
]:
    all_data[col] = all_data[col].fillna(0)

# LotFrontage: fill with median of each neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Fill remaining categorical features with mode
for col in [
    'MSZoning','Functional','Utilities','Exterior1st','Exterior2nd',
    'KitchenQual','SaleType','Electrical'
]:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Drop Utilities (no variance)
all_data = all_data.drop(['Utilities'], axis=1)


# ============================================================
# Step 5: Convert Some Numeric Columns to String (Categorical)
# ============================================================
for col in ['MSSubClass','OverallCond','YrSold','MoSold']:
    all_data[col] = all_data[col].astype(str)


# ============================================================
# Step 6: Label Encoding for Ordered Categories
# ============================================================
cols = (
    'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
    'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC',
    'BsmtExposure','BsmtFinType1','BsmtFinType2','Functional','Fence'
)
for c in cols:
    lbl = LabelEncoder()
    all_data[c] = lbl.fit_transform(all_data[c])


# ============================================================
# Step 7: Add New Feature
# ============================================================
all_data['TotalSF'] = (
    all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
)


# ============================================================
# Step 8: Fix Skewness in Numeric Features
# ============================================================
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_features = skewed_feats[abs(skewed_feats) > 0.75].index
all_data[skewed_features] = np.log1p(all_data[skewed_features])


# ============================================================
# Step 9: One-Hot Encoding
# ============================================================
all_data = pd.get_dummies(all_data)
print("Data shape after encoding:", all_data.shape)


# ============================================================
# Step 10: Split Back into Train/Test
# ============================================================
X_train = all_data[:len(train)]
X_test = all_data[len(train):]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# Step 11: Train XGBoost Model
# ============================================================

from xgboost import XGBRegressor


xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

xgb_model.fit(X_train, y)


# ============================================================
# Step 12: Evaluate Model
# ============================================================
y_pred = xgb_model.predict(X_train)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("XGBoost Performance on Training Data:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# Step 13: Predict on Test Data (Convert Back from log)
# ============================================================
preds = np.expm1(xgb_model.predict(X_test))
print("✅ Prediction completed! Sample output:")
print(preds[:10])


import pickle

# Save the model
out_path = r'/home/hardik/Desktop/python_intern/house_price.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print("✅ Model saved successfully at:", out_path)

# Load it back later
with open(out_path, 'rb') as f:
    loaded_model = pickle.load(f)