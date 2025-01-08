import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def show():
    
    df = pd.read_csv("./collegePlace.csv")

    # Define features and target
    x = df[['CGPA', 'Internships', 'Hostel', 'Age', 'HistoryOfBacklogs']]
    y = df['PlacedOrNot']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model =  DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open("./placement_rf_model.pkl", "wb") as file:
        pickle.dump(model, file)