import streamlit as st
import pandas as pd

def show():
    st.title("Feedback Form")
    st.header("Provide your valuable feedback on the Job Placement Prediction App:")

    # Collect feedback on different aspects
    feedback_on_accuracy = st.text_area("How accurate do you think the predictions were?", height=100)
    feedback_on_usability = st.text_area("How user-friendly was the application?", height=100)
    feedback_on_features = st.text_area("What features would you like to see added or improved?", height=100)
    feedback_on_appearance = st.text_area("Any suggestions for improving the appearance of the application?", height=100)
    general_feedback = st.text_area("Any other general feedback or suggestions?", height=100)

    # Get user name (optional)
    user_name = st.text_input("Enter your name (optional)")

    # Submit button
    if st.button("Submit Feedback"):
        # Create a dictionary to store feedback data
        feedback_data = {
            "User Name": user_name,
            "Accuracy Feedback": feedback_on_accuracy,
            "Usability Feedback": feedback_on_usability,
            "Features Feedback": feedback_on_features,
            "Appearance Feedback": feedback_on_appearance,
            "General Feedback": general_feedback
        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(feedback_data, index=[0]) 

        # Load existing feedback data (if any)
        try:
            existing_feedback = pd.read_csv("feedback_data.csv")
            df = pd.concat([existing_feedback, df], ignore_index=True)
        except FileNotFoundError:
            pass

        # Save feedback to CSV
        df.to_csv("feedback_data.csv", index=False) 

        st.success("Thank you for your valuable feedback!")