import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="akskhare/Tourism-Packages", filename="best_tourism-packages_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Packages App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Age = st.number_input("Age in Yrs", min_value=1.0, max_value=100.0, step=0.1)
TypeofContact = st.selectbox("Type of Contact", ['Self Enquiry','Company Contacted'])
CityTier = st.selectbox("Type of City (Tier1=1,Tier2=2, Tier3=3)", [1,2,3])
DurationOfPitch = st.number_input("Pitch Time(Hrs)", min_value=0, max_value=1000, step=1)
Occupation = st.selectbox("Customer Occupation ", ['FreeLancer','Large Business','Salaried','Small Business'])
Gender = st.selectbox("Gender", ['Male','Female'])
NumberOfPersonVisiting= st.number_input("Number of Person Visiting", min_value=1, max_value=100, step=1)
NumberOfFollowups= st.number_input("Number of Followups", min_value=1, max_value=50, step=1)
ProductPitched= st.selectbox("Product Pitched", ['Basic','Standard','Premium'])
PreferredPropertyStar= st.number_input("Preferred Property Star", min_value=1, max_value=5, step=1)
MaritalStatus= st.selectbox("Marital Status", ['Married','Single','Divorced'])
NumberOfTrips= st.number_input("Number of Trips", min_value=1, max_value=100, step=1)
Passport= st.selectbox("Passport", ['Yes','No'])
OwnCar = st.selectbox("Car Owned (Yes=1, No=0)", [1, 0])
PitchSatisfactionScore= st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, step=1)
NumberOfChildrenVisiting= st.number_input("Number of Children Visiting", min_value=0, max_value=10, step=1)
Designation = st.text_input("Designation")
MonthlyIncome= st.number_input("Monthly Income:", min_value=1,step=100)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age in Yrs': Age,
    'Type of Contact': TypeofContact,
    'Type of City (Tier1=1,Tier2=2, Tier3=3)': CityTier,
    'Pitch Time(Hrs)': DurationOfPitch,
    'Customer Occupation': Occupation,
    'Gender': Gender,
    'Number of Person Visiting': NumberOfPersonVisiting,
    'Number of Followups': NumberOfFollowups,
    'Product Pitched': ProductPitched,
    'Preferred Property Star': PreferredPropertyStar,
    'Marital Status': MaritalStatus,
    'Number of Trips': NumberOfTrips,
    'Passport': Passport,
    'Car Owned (Yes=1, No=0)': OwnCar,
    'Pitch Satisfaction Score': PitchSatisfactionScore,
    'Number of Children Visiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'Monthly Income': MonthlyIncome

}])


if st.button("ProdTaken"):
    prediction = model.predict(input_data)[0]
    result = "Product Bought" if prediction == 1 else "Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
