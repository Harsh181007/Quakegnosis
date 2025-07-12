import joblib
import numpy as np

# Load the saved model
model_filename = "quake_model.pkl"
model_reg = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

def get_user_input():
    """
    Function to get earthquake parameters from the user.
    Returns:
        Latitude (float), Longitude (float), Depth (float)
    """
    try:
        latitude = float(input("Enter Latitude: "))
        longitude = float(input("Enter Longitude: "))
        depth = float(input("Enter Depth: "))
    except ValueError:
        print("Invalid input. Please enter numeric values for Latitude, Longitude, and Depth.")
        return get_user_input()
    
    return latitude, longitude, depth

def make_prediction(lat, lon, depth):
    """
    Function to make a prediction using the trained model.
    Args:
        lat (float): Latitude of the earthquake
        lon (float): Longitude of the earthquake
        depth (float): Depth of the earthquake
    Returns:
        Predicted magnitude (float)
    """
    # Prepare input data for the model
    input_data = np.array([[lat, lon, depth]])
    
    # Make a prediction
    prediction = model_reg.predict(input_data)
    
    return prediction[0]

# Get user input for the earthquake parameters
latitude, longitude, depth = get_user_input()

# Make a prediction using the model
predicted_magnitude = make_prediction(latitude, longitude, depth)

# Display the result
print(f"\nPrediction Result:")
print(f"Latitude: {latitude}, Longitude: {longitude}, Depth: {depth}")
print(f"Predicted Magnitude: {predicted_magnitude:.2f}")
