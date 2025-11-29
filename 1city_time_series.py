import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Read CSV (use correct path)
data = pd.read_csv("/home/hardik/Desktop/python_intern/archive/City_time_series.csv")
print("Shape:", data.shape)

# Inspect data
print(data.info())
print("Described Data:\n", data.describe())
print("Columns in dataset:\n", data.columns)

# Focus on 'ZHVI_AllHomes' (exists)
print("\nData stats for ZHVI_AllHomes:")
print("Mean:", data['ZHVI_AllHomes'].mean())
print("Median:", data['ZHVI_AllHomes'].median())
print("Std:", data['ZHVI_AllHomes'].std())

# Average house price over time
avg_prices = data.groupby('Date')['ZHVI_AllHomes'].mean()
avg_prices.plot(figsize=(10, 5), title='Average House Prices Over Time')
plt.show()

print("Step 1 done.")

# Handle missing values
print("Missing values:\n", data.isnull().sum().head())
data.ffill(inplace=True)
print("Missing values handled.")

# Remove outliers based on ZHVI_AllHomes
q_low = data['ZHVI_AllHomes'].quantile(0.01)
q_high = data['ZHVI_AllHomes'].quantile(0.99)
data = data[(data['ZHVI_AllHomes'] > q_low) & (data['ZHVI_AllHomes'] < q_high)]

print("Outliers removed.")

# Normalize house prices
scaler = MinMaxScaler()
data[['ZHVI_AllHomes']] = scaler.fit_transform(data[['ZHVI_AllHomes']])

print("Normalization complete.")

# Plot histogram
sns.histplot(data['ZHVI_AllHomes'], kde=True)
plt.title("Distribution of Normalized Home Values")
plt.show()

# Optional: sample for speed
sample_data = data.sample(5000, random_state=42)

# Boxplot by region (may be large)
sns.boxplot(x='RegionName', y='ZHVI_AllHomes', data=sample_data)
plt.xticks(rotation=90)
plt.title("ZHVI_AllHomes by Region (Sample)")
plt.show()

# Correlation heatmap
corr = data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

# Extract Year
data['Year'] = pd.to_datetime(data['Date']).dt.year

# Convert StateName to dummy variables if present
if 'StateName' in data.columns:
    data = pd.get_dummies(data, columns=['StateName'], drop_first=True)

# Drop unnecessary columns
data.drop(['Date'], axis=1, inplace=True, errors='ignore')

print("Final dataset shape:", data.shape)
print("All steps completed successfully.")
