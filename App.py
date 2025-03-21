import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import base64
import warnings

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.write("### Predict your calories burned based on your personal parameters like Age, BMI, Duration, etc.")

st.sidebar.header("User Input Parameters")

# Function to collect user inputs
def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    bmi = st.sidebar.slider("BMI:", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 36, 42, 38)
    
    gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))
    model_choice = st.sidebar.selectbox("Choose Model:", ["Linear Regression", "Random Forest"])  # Default: Linear Regression
    save_results = st.sidebar.checkbox("Save Predictions")
    
    gender = 1 if gender_button == "Male" else 0  # Encoding gender (Male = 1, Female = 0)
    
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Assuming "Gender_male" exists in training data
    }
    
    features = pd.DataFrame([data_model])
    return features, model_choice, save_results

df, model_choice, save_results = user_input_features()

st.write("---")
st.header("Your Parameters")
st.write(df)

# Load Data
try:
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    
    # Add BMI Calculation
    exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)
    
    # Encode Categorical Features
    exercise_df = pd.get_dummies(exercise_df, drop_first=True)
    
    # Train-Test Split
    X = exercise_df.drop("Calories", axis=1)
    y = exercise_df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Model Selection
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3, random_state=1)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # Ensure User Input Matches Model Features
    df = df.reindex(columns=X_train.columns, fill_value=0)

    # Make Prediction
    prediction = model.predict(df)

    st.write("---")
    st.header("Prediction")
    st.success(f"🔥 You will burn approximately **{round(prediction[0], 2)} kilocalories** 🔥")

    # Model Performance
    st.write("---")
    st.header("Model Performance")
    y_pred = model.predict(X_test)
    st.write(f"📊 R² Score: {round(r2_score(y_test, y_pred), 2)}")
    st.write(f"📉 RMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}")

    # Data Visualization
    st.write("---")
    st.header("Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calories Distribution")
        fig, ax = plt.subplots()
        sn.histplot(exercise_df["Calories"], kde=True, ax=ax, color='blue')
        ax.set_title("Calories Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        sn.heatmap(exercise_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    # Download Predictions
    st.write("---")
    st.header("Download Your Prediction")
    csv = df.to_csv(index=False)
    if save_results:
        st.download_button(label="📥 Download CSV File", data=csv, file_name="prediction.csv", mime="text/csv")
    else:
        if st.button("📥 Download CSV File"):
            st.warning("Select Save Predictions and Download")

except FileNotFoundError:
    st.error("🚨 Missing dataset! Ensure `calories.csv` and `exercise.csv` are in the project folder.")

except Exception as e:
    st.error(f"🚨 Error: {str(e)}")
