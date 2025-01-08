import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def show():
    
    st.image("clgss.jpg", width=10000,)
    #st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)


    st.title("Job Journey Prediction Model Using Machine Learning")
    st.text("The Job Journey Prediction Model uses machine learning to predict students' chances of securing job placements based on academic performance, skills, and extracurricular activities. By analyzing historical data, it identifies patterns and provides actionable insights for students and institutions. The model helps students focus on areas needing improvement, assists institutions in refining training programs, and streamlines recruitment for companies. This data-driven approach enhances employability and optimizes placement processes effectively.")
    st.write("    ")
    st.write("    ")

    st.header("Machine Learning", divider="gray")
    st.text("Machine Learning is a branch of artificial intelligence that enables systems to learn and improve from experience without explicit programming. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Common applications include image recognition, natural language processing, and recommendation systems. By automating complex tasks, machine learning enhances efficiency and innovation across various industries.")
    st.write("    ")
    st.write("    ")
    st.write("    ")

    st.header("Supervised Learning")
    st.text("Supervised learning is a type of machine learning where the model is trained on labeled data, meaning the input comes with corresponding outputs. It is used to predict outcomes for new data by learning from examples, such as in classification and regression tasks.")

    st.subheader("Few Types of supervised learning", divider="grey")
    #type 1
    st.subheader("Decision Tree")
    col1, col2= st.columns(2)
    with col1:
        st.text("A decision tree is a graphical representation used for decision-making and predictive modeling. It splits data into branches based on feature conditions, leading to decisions or outcomes, making it easy to interpret.")
    with col2:
        st.image("dt.jpg")
    #type 2
    st.subheader(" Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        st.image("rf.jpg")
    with col2:
        st.text("Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and prevent overfitting. It is used for classification and regression tasks, offering robust predictions by aggregating results from various trees.")
    #type 3
    st.subheader("Logistic regression ")
    col1, col2= st.columns(2)
    with col1:
        st.text("Logistic regression is a statistical method used in machine learning to model the probability of a binary outcome (e.g., yes/no) based on input variables. It applies the logistic function to ensure predicted values lie between 0 and 1, making it ideal for classification tasks.")
    with col2:
        st.image("log.jpg")

    st.write("    ")
    st.write("    ")
    st.write("    ")

    st.header("Unupervised Learning")
    st.text("Unsupervised learning is a type of machine learning where the model analyzes unlabeled data to find hidden patterns or structures. It is commonly used for clustering, dimensionality reduction, and anomaly detection.")

    st.subheader("Few Types of unsupervised learning", divider="grey")
    #type 1
    st.subheader("K-Means")
    col1, col2 = st.columns(2)
    with col1:
        st.text("K-Means is an unsupervised machine learning algorithm used for clustering data into k groups based on similarity. It works by iteratively assigning data points to the nearest cluster centroid and updating centroids until convergence, aiming to minimize the variance within clusters.")
    with col2:
        st.image("k.png")
    #type 2
    st.subheader("Genetic Algorithm")
    col1, col2 = st.columns(2)
    with col1:
        st.image("ga.png")
    with col2:
        st.text("A Genetic Algorithm (GA) is an optimization technique inspired by natural selection, where solutions evolve over iterations. It uses operations like selection, crossover, and mutation to find optimal or near-optimal solutions to complex problems by mimicking biological evolution.")