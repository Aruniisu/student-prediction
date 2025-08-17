import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.data_processing import load_data, preprocess_data, encode_features, split_and_scale_data
from src.models import train_base_models, tune_random_forest, evaluate_model
from src.visualization import plot_numeric_distributions, plot_categorical_distributions, plot_score_relationships

# App configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Data Exploration", "Model Training", "Performance Prediction"])

# Load data
@st.cache_data
def load_and_preprocess():
    df = load_data('data/raw/StudentsPerformance.csv')
    df = preprocess_data(df)
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education']
    df, label_encoders = encode_features(df, categorical_cols)
    return df, label_encoders

df, label_encoders = load_and_preprocess()

if page == "Data Exploration":
    st.header("📊 Data Exploration")
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.dataframe(df)
    
    # Select visualization type
    viz_type = st.selectbox("Select visualization type", 
                          ["Score Distributions", "Category Distributions", "Score Relationships"])
    
    if viz_type == "Score Distributions":
        st.subheader("Score Distributions")
        numeric_cols = ['math score', 'reading score', 'writing score']
        fig = plot_numeric_distributions(df, numeric_cols)
        st.pyplot(fig)
        
    elif viz_type == "Category Distributions":
        st.subheader("Category Distributions")
        categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                           'lunch', 'test preparation course']
        selected_col = st.selectbox("Select category", categorical_cols)
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=selected_col)
        plt.title(f'Distribution of {selected_col}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_type == "Score Relationships":
        st.subheader("Score Relationships")
        score_type = st.selectbox("Select score type", ['math score', 'reading score', 'writing score'])
        category = st.selectbox("Select category", ['gender', 'race/ethnicity', 'parental level of education'])
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=category, y=score_type)
        plt.title(f'{score_type.title()} by {category}')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a few minutes"):
            # Prepare data
            features = ['gender', 'race/ethnicity', 'parental level of education', 
                       'test_prep_completed', 'standard_lunch']
            X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df, features, 'performance')
            
            # Train and evaluate models
            st.subheader("Base Model Performance")
            results, models = train_base_models(X_train, y_train)
            st.dataframe(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))
            
            st.subheader("Optimized Random Forest")
            best_model = tune_random_forest(X_train, y_train)
            
            # Evaluate and show results
            st.subheader("Evaluation Metrics")
            classification_rep = evaluate_model(best_model, X_test, y_test, "Random Forest", return_report=True)
            st.text(classification_rep)
            
            # Show feature importance
            st.subheader("Feature Importance")
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature'))
            
            # Save model and scaler
            joblib.dump(best_model, 'models/performance_predictor.pkl')
            joblib.dump(scaler, 'data/interim/scaler.pkl')
            joblib.dump(label_encoders, 'models/label_encoders.pkl')
            st.success("Model trained and saved successfully!")

elif page == "Performance Prediction":
    st.header("🔮 Performance Prediction")
    
    # Load model if available, otherwise show sample data
    try:
        model = joblib.load('models/performance_predictor.pkl')
        scaler = joblib.load('data/interim/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.info("Using sample data for demonstration. Train the model for accurate predictions.")
    
    st.subheader("Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ['male', 'female'])
        race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
        parental_edu = st.selectbox("Parental Education Level", 
                                  ['some high school', 'high school', 'some college', 
                                   "associate's degree", "bachelor's degree", "master's degree"])
    
    with col2:
        lunch = st.selectbox("Lunch Type", ['standard', 'free/reduced'])
        test_prep = st.selectbox("Test Preparation", ['none', 'completed'])
        math_score = st.slider("Math Score", 0, 100, 70)
        reading_score = st.slider("Reading Score", 0, 100, 70)
        writing_score = st.slider("Writing Score", 0, 100, 70)
    
    if st.button("Predict Performance"):
        # Prepare input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'race/ethnicity': [race],
            'parental level of education': [parental_edu],
            'lunch': [lunch],
            'test preparation course': [test_prep],
            'math score': [math_score],
            'reading score': [reading_score],
            'writing score': [writing_score]
        })
        
        # Preprocess (same steps as training)
        input_data = preprocess_data(input_data)
        for col in ['gender', 'race/ethnicity', 'parental level of education']:
            le = label_encoders[col] if model_loaded else LabelEncoder().fit(df[col])
            input_data[col] = le.transform(input_data[col])
        
        features = ['gender', 'race/ethnicity', 'parental level of education', 
                   'test_prep_completed', 'standard_lunch']
        X = input_data[features]
        
        if model_loaded:
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probas = model.predict_proba(X_scaled)[0]
        else:
            # Demo logic based on scores if model not trained
            avg_score = (math_score + reading_score + writing_score) / 3
            if avg_score < 60:
                prediction = 0  # Fail
            elif avg_score < 75:
                prediction = 1  # Pass
            elif avg_score < 90:
                prediction = 2  # Good
            else:
                prediction = 3  # Excellent
            probas = [0]*4
            probas[prediction] = 1
        
        performance_map = {0: 'Fail', 1: 'Pass', 2: 'Good', 3: 'Excellent'}
        predicted_class = performance_map[prediction]
        
        # Show results
        st.subheader("Prediction Result")
        st.success(f"Predicted Performance: {predicted_class}")
        
        # Show score visualization
        st.subheader("Student Scores")
        scores = pd.DataFrame({
            'Subject': ['Math', 'Reading', 'Writing'],
            'Score': [math_score, reading_score, writing_score]
        })
        st.bar_chart(scores.set_index('Subject'))
        
        # Show probability distribution
        st.subheader("Performance Probability Distribution")
        proba_df = pd.DataFrame({
            'Performance': ['Fail', 'Pass', 'Good', 'Excellent'],
            'Probability': probas
        })
        st.bar_chart(proba_df.set_index('Performance'))
        
        # Show performance analysis
        st.subheader("Performance Analysis")
        avg_score = (math_score + reading_score + writing_score) / 3
        st.metric("Average Score", f"{avg_score:.1f}/100")
        
        if avg_score < 60:
            st.warning("This student may need additional support")
        elif avg_score >= 90:
            st.success("Excellent performance!")