import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filepath):
    """Load raw data from CSV file"""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess student performance data:
    - Create average score
    - Create performance categories
    - Feature engineering
    - Encode categorical variables
    """
    # Create target variable
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    df['performance'] = pd.cut(df['average_score'],
                              bins=[0, 40, 60, 80, 100],
                              labels=[0, 1, 2, 3])  # 0:Fail, 1:Pass, 2:Good, 3:Excellent

    # Feature engineering
    df['test_prep_completed'] = df['test preparation course'].apply(
        lambda x: 1 if x == 'completed' else 0)
    df['standard_lunch'] = df['lunch'].apply(
        lambda x: 1 if x == 'standard' else 0)

    return df

def encode_features(df, categorical_cols):
    """Encode categorical features using LabelEncoder"""
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def split_and_scale_data(df, features, target, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_artifacts(scaler, label_encoders, X_train, X_test, y_train, y_test, features):
    """Save processed data and preprocessing artifacts"""
    os.makedirs('../data/interim', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save preprocessing objects
    joblib.dump(scaler, '../data/interim/scaler.pkl')
    for col, le in label_encoders.items():
        safe_col = col.replace('/', '_').replace(' ', '_')
        joblib.dump(le, f'../data/interim/{safe_col}_encoder.pkl')
    
    # Save processed data
    pd.DataFrame(X_train, columns=features).to_csv('../data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=features).to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)