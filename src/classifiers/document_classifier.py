import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import os
from tqdm import tqdm
import warnings
import time
from typing import List, Tuple, Dict, Any

from ..core.base_classifier import (
    DocumentClassifier as BaseDocumentClassifier, 
    ClassificationResult, 
    DocumentMetadata, 
    DocumentCategory,
    create_classification_result,
    ClassifierNotReadyError,
    InvalidDocumentError
)

warnings.filterwarnings('ignore')


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class MLDocumentClassifier(BaseDocumentClassifier):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = []
        
        # Performance tracking
        self.prediction_history = []
        self.is_trained = False
        
        # Try to load saved models
        self.load_models()
        
        # RVL-CDIP document categories 
        self.document_categories = {
            'letter': ['letter', 'correspondence', 'business_letter', 'formal_letter'],
            'form': ['form', 'application', 'registration_form', 'survey_form'],
            'email': ['email', 'mail', 'correspondence', 'communication'],
            'handwritten': ['handwritten', 'note', 'manuscript', 'written'],
            'advertisement': ['advertisement', 'ad', 'marketing', 'promotion'],
            'scientific_report': ['scientific_report', 'research_report', 'study_report', 'analysis_report'],
            'scientific_publication': ['publication', 'paper', 'article', 'research_paper'],
            'specification': ['specification', 'tech_spec', 'product_spec', 'technical_spec'],
            'file_folder': ['folder', 'directory', 'file_organizer', 'document_folder'],
            'news_article': ['news_article', 'article', 'news', 'press_article'],
            'budget': ['budget', 'financial_budget', 'budget_plan', 'budget_report'],
            'invoice': ['invoice', 'bill', 'receipt', 'payment_invoice'],
            'presentation': ['presentation', 'slides', 'deck', 'powerpoint'],
            'questionnaire': ['questionnaire', 'survey', 'poll', 'feedback_form'],
            'resume': ['resume', 'cv', 'application', 'curriculum_vitae'],
            'memo': ['memo', 'memorandum', 'internal_memo', 'business_memo']
        }
    
    # Clean and preprocess filename for feature extraction
    def preprocess_filename(self, filename):
        # Remove file extension
        name = os.path.splitext(filename)[0]
        
        # Convert to lowercase
        name = name.lower()
        
        # Replace underscores, hyphens, and dots with spaces
        name = re.sub(r'[._-]', ' ', name)
        
        # Remove numbers and special characters
        name = re.sub(r'[0-9]', '', name)
        name = re.sub(r'[^a-zA-Z\s]', '', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    # Extract various features from filenames
    def extract_features(self, filenames):
        features = []
        
        for filename in tqdm(filenames, desc="Extracting features"):
            processed_name = self.preprocess_filename(filename)
            
            # Basic features
            feature_dict = {
                'length': len(processed_name),
                'word_count': len(processed_name.split()),
                'has_numbers': bool(re.search(r'\d', filename)),
                'has_underscore': '_' in filename,
                'has_hyphen': '-' in filename,
                'has_dot': '.' in filename,
                'extension_pdf': 1 if filename.lower().endswith('.pdf') else 0,
                'extension_tif': 1 if filename.lower().endswith('.tif') else 0,
                'extension_png': 1 if filename.lower().endswith('.png') else 0,
                'extension_docx': 1 if filename.lower().endswith('.docx') else 0,
                'extension_pptx': 1 if filename.lower().endswith('.pptx') else 0,
                'extension_xlsx': 1 if filename.lower().endswith('.xlsx') else 0,
                'processed_text': processed_name
            }
            
            # Check for document category keywords
            for category, keywords in self.document_categories.items():
                feature_dict[f'contains_{category}'] = any(
                    keyword in processed_name for keyword in keywords
                )
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    # Prepare the dataset for training
    def prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        features_df = self.extract_features(df['filename'])
        
        # Combine features with labels
        df_combined = pd.concat([features_df, df['label']], axis=1)
        
        # Encode labels
        df_combined['label_encoded'] = self.label_encoder.fit_transform(df_combined['label'])
        
        return df_combined
    
    # Create text-based features using TF-IDF
    def create_text_features(self, df):
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        text_features = self.vectorizer.fit_transform(df['processed_text'])
        
        # Convert to DataFrame
        text_features_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(text_features.shape[1])]
        )
        
        return text_features_df
    
    # Train multiple traditional ML models
    def train_models(self, X_train, y_train):        
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        # Train each model
        for name, model in tqdm(models.items(), desc="Training models"):
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"{name} trained successfully")
            except Exception as e:
                print(f"{name} failed: {str(e)}")
    
    # Evaluate all trained models
    def evaluate_models(self, X_test, y_test):        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
        

        
        # Find best model
        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        self.is_trained = True
        
        print(f"\nBest model: {self.best_model_name} (Accuracy: {results[self.best_model_name]:.4f})")
        
        return results
    
    # Detailed evaluation of a specific model
    def detailed_evaluation(self, X_test, y_test, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        print(f"\nDetailed evaluation for {model_name}:")
        print("=" * 50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save trained models
    def save_models(self, output_dir='models'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, os.path.join(output_dir, 'best_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name.replace(" ", "_")}.pkl'))
        
        print(f"Models saved to {output_dir}/")
    
    # Load saved models
    def load_models(self, model_dir='models'):
        try:
            if os.path.exists(os.path.join(model_dir, 'best_model.pkl')):
                self.best_model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
                self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
                self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
                self.is_trained = True
                print(f"Loaded trained models from {model_dir}/")
            else:
                print(f"No saved models found in {model_dir}/")
        except Exception as e:
            print(f"Failed to load models: {e}")
    
    # Predict categories for new filenames
    def predict(self, filenames):
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Extract features
        features_df = self.extract_features(filenames)
        
        # Create text features using the fitted vectorizer
        text_features = self.vectorizer.transform(features_df['processed_text'])
        
        # Convert to DataFrame
        text_features_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(text_features.shape[1])]
        )
        
        # Combine features
        X = pd.concat([features_df.drop('processed_text', axis=1), text_features_df], axis=1)
        
        # Make predictions
        predictions = self.best_model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels
    
    # Interface methods for BaseDocumentClassifier - predict document category using the new interface
    def predict(self, content: str, metadata: DocumentMetadata = None) -> ClassificationResult:
        start_time = time.time()
        
        if not self.is_ready():
            raise ClassifierNotReadyError("ML classifier not trained")
        
        if not content or not content.strip():
            raise InvalidDocumentError("Empty document content")
        
        try:
            # For ML classifier, we treat content as filename for now
            # TODO: This should be adapted for actual document content
            filenames = [content]
            
            # Extract features
            features_df = self.extract_features(filenames)
            
            # Create text features using the fitted vectorizer
            text_features = self.vectorizer.transform(features_df['processed_text'])
            
            # Convert to DataFrame
            text_features_df = pd.DataFrame(
                text_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(text_features.shape[1])]
            )
            
            # Combine features
            X = pd.concat([features_df.drop('processed_text', axis=1), text_features_df], axis=1)
            
            # Make predictions with probability
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)
            
            predicted_label = self.label_encoder.inverse_transform(predictions)[0]
            confidence = float(np.max(probabilities[0]))
            
            # Create features dict for tracking
            features_used = features_df.iloc[0].to_dict()
            
            result = create_classification_result(
                category=predicted_label,
                confidence=confidence,
                classifier_type=self.classifier_type,
                start_time=start_time,
                reasoning=f"ML prediction using {self.best_model_name}",
                features_used=features_used,
                cost_estimate=0.001  # Minimal cost for ML
            )
            
            # Track prediction
            self.prediction_history.append({
                'timestamp': time.time(),
                'result': result,
                'content_length': len(content)
            })
            
            return result
            
        except Exception as e:
            raise ClassificationTimeout(f"ML classification failed: {str(e)}")
    
    # Predict multiple documents efficiently
    def predict_batch(self, documents: List[Tuple[str, DocumentMetadata]]) -> List[ClassificationResult]:
        return [self.predict(content, metadata) for content, metadata in documents]
    
    # Get current performance statistics
    def get_performance_metrics(self) -> Dict[str, float]:
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100
        
        avg_confidence = np.mean([p['result'].confidence for p in recent_predictions])
        avg_processing_time = np.mean([p['result'].processing_time_ms for p in recent_predictions])
        total_predictions = len(self.prediction_history)
        
        return {
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time,
            'total_predictions': total_predictions,
            'model_name': self.best_model_name or 'none'
        }
    
    # Check if classifier is ready to make predictions
    def is_ready(self) -> bool:
        return (self.best_model is not None and 
                self.vectorizer is not None and 
                self.label_encoder is not None and
                self.is_trained)
    
    # Return classifier type identifier
    @property
    def classifier_type(self) -> str:
        return "ML"
    
    # Legacy method for backwards compatibility - predict categories for filenames
    def predict_filenames(self, filenames):
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Extract features
        features_df = self.extract_features(filenames)
        
        # Create text features using the fitted vectorizer
        text_features = self.vectorizer.transform(features_df['processed_text'])
        
        # Convert to DataFrame
        text_features_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(text_features.shape[1])]
        )
        
        # Combine features
        X = pd.concat([features_df.drop('processed_text', axis=1), text_features_df], axis=1)
        
        # Make predictions
        predictions = self.best_model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels
    
    # Analyze feature importance for tree-based models
    def analyze_feature_importance(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = [col for col in self.feature_columns if col != 'processed_text']
            
            # Get feature importance
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        else:
            print(f"{model_name} doesn't support feature importance analysis")
            return None

# Main execution function
def main():
    # Initialize classifier
    classifier = MLDocumentClassifier()
    
    # Prepare data
    data_path = "business_documents_dataset_cleaned.csv"
    if not os.path.exists(data_path):
        print("Cleaned dataset not found. Please ensure 'business_documents_dataset_cleaned.csv' exists.")
        return
    
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
    

    
    # Detailed evaluation of best model
    classifier.detailed_evaluation(X_test, y_test)
    
    # Analyze feature importance
    classifier.analyze_feature_importance()
    
    # Save models
    classifier.save_models()
    
    # Example prediction
    test_filenames = [
        "invoice_2024_001.pdf",
        "meeting_minutes_jan_2024.docx",
        "contract_agreement_v2.pdf",
        "financial_report_q4_2024.xlsx"
    ]
    
    predictions = classifier.predict_filenames(test_filenames)
    print("\nExample predictions:")
    for filename, pred in zip(test_filenames, predictions):
        print(f"{filename} -> {pred}")

if __name__ == "__main__":
    main() 