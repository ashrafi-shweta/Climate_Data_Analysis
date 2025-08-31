import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------
# 1. Load datasets
# -------------------------
temp_data = pd.read_csv('GlobalTemperatures.csv')
co2_data = pd.read_csv('co2_emissions.csv')

# -------------------------
# 2. Process Temperature Data
# -------------------------
temp_data['Year'] = pd.to_datetime(temp_data['dt']).dt.year
yearly_temp = temp_data.groupby('Year')['LandAverageTemperature'].mean().reset_index()
yearly_temp.rename(columns={'LandAverageTemperature': 'Temperature'}, inplace=True)

# -------------------------
# 3. Process CO2 Data
# -------------------------
yearly_co2 = co2_data.groupby('year')['value'].sum().reset_index()
yearly_co2.rename(columns={'year': 'Year', 'value': 'CO2'}, inplace=True)

# -------------------------
# 4. Merge datasets
# -------------------------
merged_data = pd.merge(yearly_temp, yearly_co2, on='Year', how='inner')

# -------------------------
# 5. Visualization
# -------------------------
# Temperature Trend
plt.figure(figsize=(12,6))
sns.lineplot(x='Year', y='Temperature', data=merged_data, marker='o')
plt.title('Global Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.savefig('temperature_trend.png')
plt.show()

# CO2 Trend
plt.figure(figsize=(12,6))
sns.lineplot(x='Year', y='CO2', data=merged_data, marker='o', color='orange')
plt.title('Global CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons)')
plt.savefig('co2_trend.png')
plt.show()

# CO2 vs Temperature Scatter
plt.figure(figsize=(8,6))
sns.scatterplot(x='CO2', y='Temperature', data=merged_data)
plt.title('CO2 vs Global Temperature')
plt.xlabel('CO2 Emissions')
plt.ylabel('Temperature (°C)')
plt.savefig('co2_vs_temperature.png')
plt.show()

# -------------------------
# 6. Correlation
# -------------------------
correlation = merged_data['CO2'].corr(merged_data['Temperature'])
print(f"\nCorrelation between CO2 and Temperature: {correlation:.2f}")

# -------------------------
# 7. Linear Regression Prediction
# -------------------------
X = merged_data['CO2'].values.reshape(-1,1)
y = merged_data['Temperature'].values

model = LinearRegression()
model.fit(X, y)

# Predict future temperature for next 5 hypothetical CO2 values
future_co2 = np.array([merged_data['CO2'].max() + i*50 for i in range(1,6)]).reshape(-1,1)
predicted_temp = model.predict(future_co2)

print("\nFuture CO2 (metric tons):", future_co2.flatten())
print("Predicted Temperature (°C):", predicted_temp)

# Visualize regression prediction
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Actual Data')
plt.plot(future_co2, predicted_temp, color='red', marker='o', label='Predicted Temperature')
plt.xlabel('CO2 Emissions')
plt.ylabel('Temperature (°C)')
plt.title('Linear Regression Prediction of Temperature')
plt.legend()
plt.savefig('future_temperature_prediction.png')
plt.show()

# -------------------------
# 8. Heatmap of Temperature Variables
# -------------------------
temp_corr_data = temp_data[['LandAverageTemperature', 'LandMaxTemperature',
                           'LandMinTemperature', 'LandAndOceanAverageTemperature']].dropna()

plt.figure(figsize=(8,6))
sns.heatmap(temp_corr_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Temperature Variables")
plt.savefig('temperature_correlation_heatmap.png')
plt.show()
