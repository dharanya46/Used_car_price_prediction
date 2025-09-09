import streamlit as st
import pandas as pd
import numpy as np
import pickle
from num2words import num2words


with open("car_price_pipelinefinal2.pkl", "rb") as f:
    for_streamlit2 = pickle.load(f)

model = for_streamlit2["model"]
scaler = for_streamlit2["scaler"]
encoder = for_streamlit2["encoder"]
feature_names = for_streamlit2["feature_names"]


df = pd.read_csv("selected_data_cleaned.csv")

st.title("🚗 Used_Car Price Prediction App")


model_year = st.number_input(
    "Year of Manufacture",
    min_value=int(df["modelYear"].min()),
    max_value=int(df["modelYear"].max()),
    value=int(df["modelYear"].median()),
    step=1,
)


Km = st.number_input(
    "Kilometers Driven",
    min_value=int(df["km"].min()),
    max_value=int(df["km"].max()),
    value=int(df["km"].median()),   # default starting value
    step=1000,
)


mileage = st.number_input(
    "Mileage (km/l)",
    min_value=int(df["Mileage"].min()),
    max_value=int(df["Mileage"].max()),
    value=int(df["Mileage"].median()),   # default starting value
    step=1,
)

#mileage = st.slider(
#    "Mileage (km/l)",
#    min_value=0.0,
#    max_value=50.0,
#    value=15.0,   # default starting value
#    step=0.5,
#)




owner_no = st.number_input(
    "Number of Owners",
    min_value=int(df["ownerNo"].min()),
    max_value=int(df["ownerNo"].max()),
    value=int(df["ownerNo"].median()),
    step=1,
)

body_type = st.selectbox("Body Type", sorted(df["body_type"].unique()))
oem = st.selectbox("Car Brand (OEM)", sorted(df["oem"].unique()))
transmission = st.selectbox("Transmission Type", sorted(df["transmission"].unique()))
steering = st.selectbox("Steering Type", sorted(df["Miscellaneous_Steering Type"].unique()))


if st.button("Predict Price"):
    
    body_type_encoded = encoder.transform([body_type])[0]

    
    input_data = {
        "modelYear": [model_year],
        "km": [Km],
        "Mileage": [mileage],
        "ownerNo": [owner_no],
        "body_type": [body_type_encoded],
        f"oem_{oem}": [1],
        f"transmission_{transmission}": [1],
        f"Miscellaneous_Steering Type_{steering}": [1],
    }

    input_df = pd.DataFrame(input_data)

   
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    
    numeric_cols = ["modelYear", "km", "Mileage", "ownerNo"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    
    prediction_log = model.predict(input_df)[0]
    prediction = prediction_log 
    #prediction = np.expm1(prediction_log)  

    prediction_words = num2words(int(prediction), lang='en_IN').replace(",", "").title()
    prediction_words = prediction_words + " Rupees"

    
    st.success(f"💰 Estimated Car Price: **₹ {prediction:,.2f}**")
    st.write(f"🔠 In Words: **{prediction_words}**")
