import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

def show():


    #Define the title of the app
    st.title("Research")
    st.subheader("Comparison of Machine Learning Algorithms")
    st.write("This involves comparing the performance of four machine learning algorithms—Logistic Regression, Random Forest, Decision Tree, and Support Vector Machine (SVM)—using the dataset given below. Each algorithm will be evaluated based on metrics such as accuracy, precision and recall to determine its effectiveness in predicting placement outcomes. This comparison will help identify the most suitable model for accurate and reliable predictions.")
    # Load and display the dataset
    st.header("Dataset", divider="grey")
    uploaded_file = pd.read_csv("./collegePlace.csv")

    if uploaded_file is not None:
        df = uploaded_file
        st.write("Preview of the Dataset:")
        st.dataframe(df)

        sf = df[["Age", "Internships", "CGPA","Hostel", "HistoryOfBacklogs"]]

        # Feature and target columns
        st.sidebar.header("Feature and Target Selection")
        features = st.sidebar.multiselect("Select Features", sf.columns, default=sf.columns[:-1])
        target = st.sidebar.selectbox("Select Target", df.columns, index=len(df.columns) - 1)

        if features and target:
            # Split the dataset into training and testing sets
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define algorithms
            algorithms = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Support Vector Machine": SVC(probability=True)
            }

            # Store metrics
            metrics = {
                "Algorithm": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": []
            }

            # Train and evaluate each algorithm
            for name, model in algorithms.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')

                metrics["Algorithm"].append(name)
                metrics["Accuracy"].append(accuracy)
                metrics["Precision"].append(precision)
                metrics["Recall"].append(recall)

            # Convert metrics to a DataFrame
            metrics_df = pd.DataFrame(metrics)

            st.header("Performance Metrics")
            st.write("   ")
            st.dataframe(metrics_df,use_container_width=True)
            st.header("Comparison chart")

            sns.set_theme(style="darkgrid", rc={"axes.facecolor": "black", "figure.facecolor": "black"})
            chart_type = st.selectbox("Select Metric That you want to Visualize", ["Accuracy", "Precision", "Recall"])
            st.write("")
            st.write("")
            st.write("")

            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=metrics, x="Algorithm", y=chart_type, palette="cool")

            # Customize plot appearance
            plt.title(f"Comparison of {chart_type}", fontsize=16, color="white")
            plt.xlabel("Algorithm", fontsize=14, color="white")
            plt.ylabel(chart_type, fontsize=14, color="white")
            plt.xticks(rotation=45, color="white")
            plt.yticks(color="white")
            plt.tight_layout()

            # D
            st.pyplot(plt)
