import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving the model

# Load the dataset
df_load = pd.read_csv("database.csv")

# Remove all fields we don't need
lst_dropped_columns = ['Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 
                       'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 
                       'Horizontal Error', 'Root Mean Square', 'Source', 'Location Source', 
                       'Magnitude Source', 'Status']
df_load = df_load.drop(columns=lst_dropped_columns)

# Create a 'Year' field from the 'Date' column
df_load['Date'] = pd.to_datetime(df_load['Date'], format='%d/%m/%Y', errors='coerce')
df_load['Year'] = df_load['Date'].dt.year

# Drop rows with null values
df_load = df_load.dropna()

# Create the quakes frequency dataframe
df_quake_freq = df_load.groupby('Year').size().reset_index(name='Counts')

# Calculate max and avg magnitude per year
df_max = df_load.groupby('Year')['Magnitude'].max().reset_index(name='Max_Magnitude')
df_avg = df_load.groupby('Year')['Magnitude'].mean().reset_index(name='Avg_Magnitude')

# Merge the max and avg DataFrames with df_quake_freq
df_quake_freq = df_quake_freq.merge(df_avg, on='Year').merge(df_max, on='Year')

# Load the test dataset
df_test = pd.read_csv("query.csv")

# Clean and rename the test dataset columns
df_test_clean = df_test[['time', 'latitude', 'longitude', 'mag', 'depth']].copy()
df_test_clean = df_test_clean.rename(columns={
    'time': 'Date', 'latitude': 'Latitude', 'longitude': 'Longitude', 
    'mag': 'Magnitude', 'depth': 'Depth'
})

# Convert columns to appropriate data types
df_test_clean[['Latitude', 'Longitude', 'Depth', 'Magnitude']] = df_test_clean[['Latitude', 'Longitude', 'Depth', 'Magnitude']].astype(float)

# Prepare training and testing datasets
df_training = df_load[['Latitude', 'Longitude', 'Depth', 'Magnitude']]
df_testing = df_test_clean[['Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Split training data into features and labels
X_train = df_training[['Latitude', 'Longitude', 'Depth']]
y_train = df_training['Magnitude']
X_test = df_testing[['Latitude', 'Longitude', 'Depth']]

# Train the RandomForest model
model_reg = RandomForestRegressor()
model_reg.fit(X_train, y_train)

# Save the trained model to a file
model_filename = "quake_model.pkl"
joblib.dump(model_reg, model_filename)
print(f"Model saved to {model_filename}")

# Make predictions on the test set
predictions = model_reg.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(df_testing['Magnitude'], predictions))
print(f"Root Mean Square Error on Test data = {rmse:.2f}")

# Create a prediction dataframe
df_pred_results = df_testing[['Latitude', 'Longitude']].copy()
df_pred_results['Pred_Magnitude'] = predictions
df_pred_results['Year'] = 2017
df_pred_results['RMSE'] = rmse

# Show prediction results
print("Prediction Results DataFrame:")
print(df_pred_results.head())
