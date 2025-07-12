import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the entire dataset
df = pd.read_csv("database.csv")

# Convert necessary columns to numeric and handle missing values
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')

# Drop rows with missing values to ensure plotting is smooth
df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Magnitude'])

# Latitude vs Magnitude
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Latitude', y='Magnitude', hue='Magnitude', palette='coolwarm', data=df)
plt.title('Latitude vs. Earthquake Magnitude')
plt.xlabel('Latitude')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend(title='Magnitude', bbox_to_anchor=(1, 1))
plt.show()

# Longitude vs Magnitude with trend line
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Longitude', y='Magnitude', hue='Magnitude', palette='coolwarm', data=df)
sns.regplot(x='Longitude', y='Magnitude', data=df, scatter=False, color='gray', ci=None)  # Adding trend line
plt.title('Longitude vs. Earthquake Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend(title='Magnitude', bbox_to_anchor=(1, 1))
plt.show()

# Depth vs Magnitude with size for depth
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Depth', y='Magnitude', size='Depth', sizes=(20, 200), hue='Magnitude', palette='coolwarm', data=df)
plt.title('Depth vs. Earthquake Magnitude')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend(title='Magnitude', bbox_to_anchor=(1, 1))
plt.show()
