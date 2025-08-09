import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from ..classifiers.document_classifier import MLDocumentClassifier

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and analyze the cleaned dataset"""    
    df = pd.read_csv('business_documents_dataset_cleaned.csv')
    return df

def create_data_visualizations(df):
    """Create visualizations for the dataset analysis"""
    os.makedirs('visualizations', exist_ok=True)
    #Document Type Distribution
    plt.figure(figsize=(14, 8))
    label_counts = df['label'].value_counts()
    
    # Create bar plot
    ax = sns.barplot(x=label_counts.values, y=label_counts.index, palette='viridis')
    
    # Add value labels on bars
    for i, v in enumerate(label_counts.values):
        ax.text(v + 10, i, str(v), va='center', fontweight='bold')
    
    plt.title('Document Type Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Documents', fontsize=12)
    plt.ylabel('Document Type', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/document_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  
    
    #File Extension Analysis
    df['extension'] = df['filename'].apply(lambda x: os.path.splitext(x)[1].lower())
    extension_counts = df['extension'].value_counts().head(8)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(extension_counts)))
    plt.pie(extension_counts.values, labels=extension_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('File Extension Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('visualizations/file_extension_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def run_model_evaluation_and_visualize():
    """Run the document classifier and create model evaluation visualizations"""
    
    # Initialize and run the classifier
    classifier = DocumentClassifier()
    
    # Prepare data
    data_path = "business_documents_dataset_cleaned.csv"
    df = classifier.prepare_data(data_path)
    
    # Create text features
    text_features_df = classifier.create_text_features(df)
    
    # Prepare final feature set
    feature_columns = [col for col in df.columns if col not in ['label', 'label_encoded', 'processed_text']]
    X = pd.concat([df[feature_columns], text_features_df], axis=1)
    y = df['label_encoded']
    
    # Store feature columns for later use
    classifier.feature_columns = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(classifier.label_encoder.classes_)}")
    
    # Train models
    classifier.train_models(X_train, y_train)
    
    # Evaluate models
    results = classifier.evaluate_models(X_test, y_test)
    
    # Create model evaluation visualizations
    create_model_evaluation_visualizations(classifier, X_train, X_test, y_train, y_test, results)
    
    return classifier, results

def create_model_evaluation_visualizations(classifier, X_train, X_test, y_train, y_test, results):
    """Create visualizations for model evaluation results"""
    #Model Performance Comparison
    plt.figure(figsize=(12, 8))
    model_names = list(results.keys())
    accuracies = list(results.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #Confusion Matrix for Best Model
    best_model_name = classifier.best_model_name
    best_model = classifier.best_model
    
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classifier.label_encoder.classes_,
               yticklabels=classifier.label_encoder.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #Training vs Test Performance
    train_scores = {}
    test_scores = {}
    
    for name, model in classifier.models.items():
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        train_scores[name] = train_score
        test_scores[name] = test_score
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(train_scores))
    width = 0.35
    
    plt.bar(x - width/2, list(train_scores.values()), width, label='Training Accuracy', 
           alpha=0.8, color='skyblue', edgecolor='black')
    plt.bar(x + width/2, list(test_scores.values()), width, label='Test Accuracy', 
           alpha=0.8, color='lightcoral', edgecolor='black')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training vs Test Performance', fontsize=16, fontweight='bold')
    plt.xticks(x, list(train_scores.keys()), rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/training_vs_test_performance.png', dpi=300, bbox_inches='tight')
    plt.close()



def main():
    """Main function to run all visualizations"""
    print("Document Classification Visualization Report \n")
    
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Create data visualizations
    df = create_data_visualizations(df)
    
    # Run model evaluation and create visualizations
    classifier, results = run_model_evaluation_and_visualize()
    
if __name__ == "__main__":
    main() 