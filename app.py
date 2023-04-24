import streamlit as st
import pandas as pd
import tensorflow as tf

# Load the trained model and the scaler
import joblib

# Load the scaler object
scaler = joblib.load('scaler.pkl')
# Load the model
model = tf.keras.models.load_model("model.h5")
num_cols = ['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT']


def predict_price(input_data):
    return model.predict(input_data)
st.title("Housing Price Predictor")

# Get user input
beds = st.number_input("Number of Bedrooms", min_value=1, max_value=10)
baths = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
sqft = st.number_input("Square Feet", min_value=500, max_value=10000)
lot_size = st.number_input("Lot Size", min_value=500, max_value=100000)
year_built = st.number_input("Year Built", min_value=1900, max_value=2023)

# Dummy variable input
property_types = ['PROPERTY_TYPE_Single Family Residential', 'PROPERTY_TYPE_Condo', 'PROPERTY_TYPE_Townhouse']
property_type = st.selectbox('Property Type', property_types)

# Create a dataframe from the input data
input_data = pd.DataFrame({
    "BEDS": [beds],
    "BATHS": [baths],
    "SQUARE FEET": [sqft],
    "LOT SIZE": [lot_size],
    "YEAR BUILT": [year_built],
    property_type: [1],
})

# Fill zeros for other property types
for pt in property_types:
    if pt != property_type:
        input_data[pt] = [0]

# Preprocess the input data
input_data[num_cols] = scaler.transform(input_data[num_cols])

# Make predictions
predicted_price = predict_price(input_data)

# Display predictions
st.write(f"Predicted Price: ${predicted_price[0][0]:,.2f}")
