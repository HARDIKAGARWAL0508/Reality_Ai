import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("/home/hardik/Desktop/python_intern/house-prices-advanced-regression-techniques/train.csv")

train_df.head()

print(train_df.describe().T)

print("--- Descriptive Statistics for Categorical Columns ---")
print(train_df.describe(include=['object']).T)


test_df=pd.read_csv("/home/hardik/Desktop/python_intern/house-prices-advanced-regression-techniques/test.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

print("--- SalePrice Statistics ---")
print(train_df['SalePrice'].describe())


plt.figure(figsize=(10, 5))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Log-transform the target variable
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

plt.figure(figsize=(10, 5))
sns.histplot(train_df['SalePrice'], kde=True, color='green')
plt.title('Log-Transformed Distribution of SalePrice')
plt.show()

# Store IDs and SalePrice, then drop them from the main dataframes
train_id = train_df['Id']
test_id = test_df['Id']
y_train = train_df['SalePrice']

train_df.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)

# Combine train and test data for easier preprocessing
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

train_df.isnull().sum()

print(all_data['LotFrontage'].isnull().sum())

# Impute 'None' for categorical features where NA means absence
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'):
    all_data[col] = all_data[col].fillna('None')

# Impute 0 for numerical features where NA means absence
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)

# Impute LotFrontage with the median of the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Impute with the most frequent value (mode) for the rest
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Drop Utilities as it has very little variance
all_data = all_data.drop(['Utilities'], axis=1)

print("Missing values handled!")

all_data.isnull().sum()

# Some numerical features are actually categories, so convert them to strings
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Label Encode categorical features that have a clear order
cols_to_encode = (
    'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'LandSlope',
    'LotShape', 'PavedDrive', 'Street', 'Alley', 'OverallCond'
)
for c in cols_to_encode:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# Add a feature for total square footage
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

print("Feature transformation and engineering complete.")

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

print(f"{len(skewed_feats)} skewed numerical features were log-transformed.")

all_data = pd.get_dummies(all_data)
print(f"Data shape after one-hot encoding: {all_data.shape}")

X_train = all_data[:len(train_df)]
X_test = all_data[len(train_df):]

print("--- Preprocessing Complete ---")
print(f"Shape of final training data (X_train): {X_train.shape}")
print(f"Shape of final test data (X_test): {X_test.shape}")
print(f"Shape of target variable (y_train): {y_train.shape}")



sns.histplot(y_train, kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# If you want to apply a log transformation, you would do it on y_train
# y_train_log = np.log1p(y_train)

# 1. Select only the columns with numerical data types
numerical_df = train_df.select_dtypes(include=np.number)

# 2. Now, create the correlation matrix from the numerical-only DataFrame
corrmat = numerical_df.corr()

# 3. Plot the heatmap
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# Plot a feature from the training set against the target variable
plt.scatter(x=train_df['GrLivArea'], y=y_train)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()