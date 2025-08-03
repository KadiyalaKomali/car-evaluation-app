import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load encoders
encoders = {}
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
for col in columns:
    encoders[col] = pickle.load(open(f"le_{col}.pkl", "rb"))

le_target = pickle.load(open("le_target.pkl", "rb"))

st.set_page_config(page_title="🚗 Car Evaluation App")
st.title("🚗 Car Evaluation Using Random Forest")

with st.form("car_form"):
    buying = st.selectbox("Buying Price", encoders["buying"].classes_)
    maint = st.selectbox("Maintenance Cost", encoders["maint"].classes_)
    doors = st.selectbox("Number of Doors", encoders["doors"].classes_)
    persons = st.selectbox("Capacity (Persons)", encoders["persons"].classes_)
    lug_boot = st.selectbox("Luggage Boot Size", encoders["lug_boot"].classes_)
    safety = st.selectbox("Safety", encoders["safety"].classes_)

    submit = st.form_submit_button("Predict Evaluation")

if submit:
    input_data = pd.DataFrame({
        "buying": [encoders["buying"].transform([buying])[0]],
        "maint": [encoders["maint"].transform([maint])[0]],
        "doors": [encoders["doors"].transform([doors])[0]],
        "persons": [encoders["persons"].transform([persons])[0]],
        "lug_boot": [encoders["lug_boot"].transform([lug_boot])[0]],
        "safety": [encoders["safety"].transform([safety])[0]],
    })

    pred = model.predict(input_data)[0]
    result = le_target.inverse_transform([pred])[0]
    
    st.success(f"Predicted Car Evaluation: **{result.upper()}**")
