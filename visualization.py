import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numeric_distributions(df, numeric_cols, save_path=None):
    """Plot distributions of numeric columns"""
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_categorical_distributions(df, categorical_cols, save_dir=None):
    """Plot distributions of categorical columns"""
    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        
        if save_dir:
            plt.savefig(f'{save_dir}/{col}_distribution.png')
            plt.close()
        else:
            plt.show()

def plot_score_relationships(df, categorical_cols, target='math score', save_dir=None):
    """Plot relationships between categorical features and scores"""
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=col, y=target)
        plt.title(f'{target.title()} Score by {col}')
        plt.xticks(rotation=45)
        
        if save_dir:
            safe_col = col.replace('/', '_')
            plt.savefig(f'{save_dir}/{target}_by_{safe_col}.png')
            plt.close()
        else:
            plt.show()

def plot_correlation_matrix(df, numeric_cols, save_path=None):
    """Plot correlation matrix for numeric columns"""
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Score Correlation Heatmap')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(results, save_path=None):
    """Plot comparison of model performance"""
    plt.figure(figsize=(10, 6))
    pd.Series(results).sort_values().plot(kind='barh')
    plt.title('Model Comparison (5-fold CV Accuracy)')
    plt.xlabel('Accuracy')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()