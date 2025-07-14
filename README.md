# Business Document Classification System

A comprehensive machine learning system for classifying business documents using traditional ML models based on filename analysis. This system is designed for enterprise data warehousing and retrieval applications.

## Overview

This project implements a document classification system that can categorize documents into 15 meaningful categories based on business document analysis:
- **Letter** - Business letters, correspondence
- **Form** - Applications, registration forms, surveys
- **Email** - Email communications, messages
- **Handwritten** - Handwritten notes, manuscripts
- **Advertisement** - Marketing materials, ads, promotions
- **Scientific Report** - Research reports, study reports, analysis
- **Scientific Publication** - Academic papers, journal articles, conference papers
- **Specification** - Technical specifications, product specs
- **File Folder** - Document folders, directories, organizers
- **News Article** - News articles, press releases
- **Budget** - Financial budgets, budget plans
- **Invoice** - Billing documents, receipts, payment records
- **Presentation** - Slides, decks, PowerPoint files
- **Questionnaire** - Surveys, polls, feedback forms
- **Resume** - CVs, job applications, curriculum vitae
- **Memo** - Business memos, memorandums

## Features

- **Multiple Traditional ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Naive Bayes
- **Comprehensive Feature Engineering**: TF-IDF, document keyword detection, filename pattern analysis
- **15 Document Categories**: Business document classification based on filename analysis
- **Data Visualization**: Comprehensive charts and plots for analysis
- **Easy Deployment**: Simple API for integration into existing systems
- **Detailed Evaluation**: Comprehensive performance analysis and visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

The project includes a pre-cleaned dataset with 15,000 samples across 15 document categories:

```bash
# The dataset is already included: business_documents_dataset_cleaned.csv
# Contains 15 document categories with 1000 samples each
```

### 2. Model Training

Train the document classification models:

```bash
python document_classifier.py
```

This will:
- Extract features from filenames
- Train multiple ML models
- Evaluate performance
- Save the best model

### 3. Model Evaluation and Visualization

Run the evaluation and visualization scripts:

```bash
python model_evaluation.py
python visualizations.py
```

This will:
- Load trained models
- Generate performance comparisons
- Create comprehensive visualizations
- Generate evaluation reports

## Model Performance

The system trains and evaluates multiple traditional ML models:
- Random Forest
- Gradient Boosting  
- Logistic Regression
- SVM
- Naive Bayes

Run the evaluation scripts to see actual performance metrics for your specific dataset.

## File Structure

```
data-analysis/
├── document_classifier.py              # Main classification system
├── visualizations.py                   # Data and model visualization scripts
├── model_evaluation.py                 # Model evaluation and analysis
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── models/                             # Saved trained models
├── visualizations/                     # Generated charts and plots
├── business_documents_dataset_cleaned.csv  # Main dataset (15,000 samples)
└── ds/                                 # Additional datasets
```

## API Usage

After training, you can use the models programmatically:

```python
from document_classifier import DocumentClassifier
import joblib

# Load trained model
classifier = DocumentClassifier()
classifier.best_model = joblib.load('models/best_model.pkl')
classifier.vectorizer = joblib.load('models/vectorizer.pkl')
classifier.label_encoder = joblib.load('models/label_encoder.pkl')

# Predict document categories
filenames = [
            "invoice_2024_001.pdf",
        "letter_client_a_2024.docx",
        "scientific_report_research_2024.pdf",
        "presentation_q1_results.pptx"
]

predictions = classifier.predict(filenames)
for filename, pred in zip(filenames, predictions):
    print(f"{filename} -> {pred}")
```

## Business Applications

This system is designed for enterprise use cases:

### Data Warehousing
- Automatically categorize documents during ingestion
- Improve search and retrieval capabilities
- Enable better data governance

### Document Management
- Organize document repositories
- Implement automated filing systems
- Reduce manual classification effort

### Compliance and Audit
- Identify sensitive document types
- Ensure proper document handling
- Support regulatory requirements

### Workflow Automation
- Route documents to appropriate teams
- Trigger automated processes based on document type
- Improve business process efficiency

## Technical Details

### Feature Engineering

The system extracts various features from filenames:

1. **Text Features**: TF-IDF vectorization of processed filenames
2. **Pattern Features**: Detection of numbers, special characters, extensions
3. **Business Keywords**: Category-specific keyword matching
4. **Structural Features**: Length, word count, naming patterns

### Model Selection

Multiple traditional ML models are trained and compared:
- **Random Forest**: Robust, handles non-linear relationships
- **Gradient Boosting**: High performance, good generalization
- **Logistic Regression**: Interpretable, fast inference
- **SVM**: Good for high-dimensional data
- **Naive Bayes**: Fast, works well with text data

### Evaluation Metrics

Comprehensive evaluation using:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Cross-validation
- Learning curves
- Feature importance analysis

## Customization

### Adding New Categories

To add new document categories:

1. Update the `document_categories` dictionary in `document_classifier.py`
2. Add category-specific keywords
3. Retrain the models

### Custom Datasets

To use your own dataset:

1. Prepare a CSV file with `filename` and `label` columns
2. Update the data loading code in `data_preparation.py`
3. Ensure labels match the business categories

## Performance Optimization

### For Large Datasets
- Use batch processing for feature extraction
- Implement parallel training for multiple models
- Consider using sparse matrices for memory efficiency

### For Production Deployment
- Use model serialization for fast loading
- Implement caching for repeated predictions
- Add logging and monitoring capabilities

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_features` in TF-IDF vectorizer
2. **Slow Training**: Use smaller datasets for initial testing
3. **Low Accuracy**: Check data quality and label consistency

### Getting Help

- Check the evaluation reports for model performance insights
- Review feature importance analysis for data quality issues
- Ensure proper data preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research or business applications, please cite:

```bibtex
@article{business_document_classification_2024,
  title={Business Document Classification Using Traditional Machine Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the repository or contact the maintainers. 