from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_base_models(X_train, y_train):
    """Train and evaluate base models using cross-validation"""
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = cv_scores.mean()
        print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, models

def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and generate visualizations"""
    y_pred = model.predict(X_test)
    
    # Classification report
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Fail', 'Pass', 'Good', 'Excellent'],
               yticklabels=['Fail', 'Pass', 'Good', 'Excellent'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'../reports/figures/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        features = X_test.columns if hasattr(X_test, 'columns') else range(len(importances))
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title(f'{model_name} Feature Importances')
        plt.savefig(f'../reports/figures/{model_name}_feature_importance.png')
        plt.close()

def save_model(model, filepath):
    """Save trained model to file"""
    joblib.dump(model, filepath)