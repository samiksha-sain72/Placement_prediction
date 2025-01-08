import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def show():

    # Streamlit Title
    st.title("Prediction Input Form")

    st.subheader("Predicting placement of the Students based upon their CGPA, Internships, Backlogs etc.")
    st.write("")
    st.write("")
    st.write("")

    # Input Widgets
    cgpa = st.slider("Select your CGPA(Cumulative Grade Point Average) :", min_value=0.0, max_value=10.0, value=7.5, step=0.1)  # Slider for CGPA
    internships = st.number_input("Select the Number of Internships that you had :", min_value=0, max_value=10, value=1, step=1)
    hostel = st.selectbox("Hostel :", ["Yes", "No"])  # Dropdown for Hostel
    age = st.slider("Age :", min_value=18, max_value=30, value=22, step=1)  # Slider for Age
    backlogs = st.number_input("History of Backlogs :", min_value=0, max_value=1, value=0, step=1)

    # Process categorical inputs (e.g., Hostel: Yes/No -> 1/0)
    hostel_map = {"Yes": 1, "No": 0}
    hostel_encoded = hostel_map[hostel]

    # Convert inputs into a DataFrame
    input_data = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Hostel": [hostel_encoded],
        "Age": [age],
        "HistoryOfBacklogs": [backlogs]
    })

    # Display the DataFrame (optional for debugging)
    st.header("Input DataFrame:")# put this in a header write change kardena header taki thoda bda ho jaye -------------------
    st.dataframe(input_data,use_container_width=True)

    # Placeholder for predictions
    if st.button("Predict"):
        # Load the trained model (replace 'your_model.pkl' with your model file path)
        import pickle
        with open("./placement_rf_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Make predictions using the input data
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display the result
        if prediction == 1:
            st.success(f"The student is likely to be placed! 🎉\nProbability: {probability:.2f}")
            st.balloons()
        else:
            st.error(f"The student is unlikely to be placed.\nProbability: {probability:.2f}")
            # st.snow()