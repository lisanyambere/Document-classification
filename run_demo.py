#!/usr/bin/env python3
"""
Demo script for Document Classification System
Shows the complete system in action
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_requirements():
    """Check if system is ready to run"""
    issues = []
    
    # Check .env file
    if not Path('.env').exists():
        issues.append("Missing .env file (run: python setup.py)")
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key or api_key == 'your-groq-api-key-here':
        issues.append("GROQ_API_KEY not set in .env file")
    
    # Check dataset
    if not Path('business_documents_dataset_cleaned.csv').exists():
        issues.append("Missing training dataset: business_documents_dataset_cleaned.csv")
    
    # Check models directory
    if not Path('models').exists():
        issues.append("Missing models directory (run: python setup.py)")
    
    return issues


def demo_ml_classifier():
    """Demo ML classifier functionality"""
    print("\n" + "="*50)
    print("DEMO: ML Classifier")
    print("="*50)
    
    try:
        from document_classifier import MLDocumentClassifier
        from base_classifier import DocumentMetadata
        
        print("Initializing ML classifier...")
        classifier = MLDocumentClassifier()
        
        if not classifier.is_ready():
            print("ML classifier not trained. Training required first.")
            print("Run: python document_classifier.py")
            return False
        
        # Test documents
        test_docs = [
            ("Invoice #123 for consulting services", DocumentMetadata(file_extension="pdf")),
            ("Technical report on algorithms", DocumentMetadata(file_extension="docx")),
            ("Business letter to client", DocumentMetadata(file_extension="txt")),
        ]
        
        print("\nClassifying test documents...")
        for i, (content, metadata) in enumerate(test_docs, 1):
            try:
                result = classifier.predict(content, metadata)
                print(f"{i}. '{content[:30]}...'")
                print(f"   -> {result.category.value} ({result.confidence:.3f} confidence)")
                print(f"   -> {result.processing_time_ms:.1f}ms processing time")
            except Exception as e:
                print(f"{i}. Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"ML Classifier demo failed: {e}")
        return False


def demo_llm_classifier():
    """Demo LLM classifier functionality"""
    print("\n" + "="*50)
    print("DEMO: LLM Classifier")
    print("="*50)
    
    try:
        from llm_classifier import LLMDocumentClassifier
        from base_classifier import DocumentMetadata
        
        print("Initializing LLM classifier...")
        classifier = LLMDocumentClassifier()
        
        if not classifier.is_ready():
            print(" LLM classifier not ready. Check GROQ_API_KEY in .env file")
            return False
        
        print(f" Using model: {classifier.model_name}")
        print(f" Max chars: {classifier.max_chars}")
        
        # Test complex document
        complex_doc = """
        TECHNICAL SPECIFICATION DOCUMENT
        
        Product: Advanced Machine Learning Pipeline
        Version: 2.1.0
        
        OVERVIEW:
        This document outlines the technical specifications for implementing
        an advanced machine learning pipeline for document classification.
        
        ARCHITECTURE:
        - Microservices design with API gateway
        - Real-time processing capabilities
        - Horizontal scaling support
        - Circuit breaker patterns for resilience
        
        PERFORMANCE REQUIREMENTS:
        - Processing speed: >1000 docs/minute
        - Accuracy: >95% on benchmark datasets
        - Latency: <100ms for standard documents
        - Cost efficiency: <$0.01 per classification
        
        SECURITY FEATURES:
        - End-to-end encryption
        - API authentication and rate limiting
        - Audit logging and monitoring
        - Data anonymization compliance
        """
        
        metadata = DocumentMetadata(
            sender_email="architect@company.com",
            file_extension="pdf",
            file_size=25600
        )
        
        print(f"\nClassifying complex technical document ({len(complex_doc)} chars)...")
        start_time = time.time()
        
        result = classifier.predict(complex_doc, metadata)
        
        print(f" Category: {result.category.value}")
        print(f" Confidence: {result.confidence:.3f}")
        print(f" Processing time: {result.processing_time_ms:.0f}ms")
        print(f" Reasoning: {result.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        print(f" LLM Classifier demo failed: {e}")
        return False


def demo_intelligent_routing():
    """Demo the full intelligent routing system"""
    print("\n" + "="*50)
    print(" DEMO: Intelligent Routing System")
    print("="*50)
    
    try:
        from document_classifier import MLDocumentClassifier
        from llm_classifier import LLMDocumentClassifier
        from agent import SupervisoryAgent
        from base_classifier import DocumentMetadata
        
        print("Initializing complete system...")
        
        # Initialize classifiers
        ml_classifier = MLDocumentClassifier()
        llm_classifier = LLMDocumentClassifier()
        
        # Check readiness
        ml_ready = ml_classifier.is_ready()
        llm_ready = llm_classifier.is_ready()
        
        print(f"ML Classifier: {' Ready' if ml_ready else ' Not trained'}")
        print(f"LLM Classifier: {' Ready' if llm_ready else ' API issue'}")
        
        if not (ml_ready or llm_ready):
            print(" Neither classifier is ready. Cannot demo routing.")
            return False
        
        # Create supervisory agent
        agent = SupervisoryAgent(ml_classifier, llm_classifier)
        print(" Supervisory agent initialized")
        
        # Test documents with different complexity levels
        test_documents = [
            {
                "name": "Simple Invoice",
                "content": "Invoice #001 for services rendered. Amount: $1,500. Due: 30 days.",
                "metadata": DocumentMetadata(file_extension="pdf"),
                "expected_route": "ML"
            },
            {
                "name": "Complex Technical Document", 
                "content": """
                METHODOLOGY FOR IMPLEMENTING ADVANCED NEURAL ARCHITECTURES
                
                This research paper presents a novel approach to implementing 
                transformer-based architectures for multi-modal document analysis.
                Our methodology incorporates attention mechanisms with positional
                encodings to achieve state-of-the-art performance on benchmark datasets.
                
                The proposed framework utilizes a hybrid approach combining:
                - Convolutional neural networks for spatial feature extraction
                - Recurrent networks for temporal sequence modeling  
                - Transformer attention for global context understanding
                - Regularization techniques for improved generalization
                """,
                "metadata": DocumentMetadata(
                    sender_email="researcher@university.edu",
                    file_extension="pdf"
                ),
                "expected_route": "LLM"
            },
            {
                "name": "Business Letter",
                "content": "Dear Mr. Johnson, Thank you for your inquiry about our services. We would be happy to discuss your project requirements.",
                "metadata": DocumentMetadata(file_extension="txt"),
                "expected_route": "ML"
            }
        ]
        
        print(f"\nTesting routing with {len(test_documents)} documents...")
        
        results = []
        for i, doc in enumerate(test_documents, 1):
            print(f"\n[{i}] {doc['name']}")
            print(f"Content: {doc['content'][:60]}...")
            
            try:
                start_time = time.time()
                result = agent.classify_document(doc['content'], doc['metadata'])
                total_time = (time.time() - start_time) * 1000
                
                # Extract routing info
                routing_decision = result.features_used.get('routing_decision', 'Unknown')
                predicted_confidence = result.features_used.get('predicted_ml_confidence', 0)
                
                print(f" Result: {result.category.value}")
                print(f" Confidence: {result.confidence:.3f}")
                print(f" Routed to: {routing_decision} classifier")
                print(f" Predicted ML confidence: {predicted_confidence:.3f}")
                print(f" Total time: {total_time:.1f}ms")
                
                results.append({
                    'name': doc['name'],
                    'routing': routing_decision,
                    'category': result.category.value,
                    'confidence': result.confidence,
                    'time_ms': total_time
                })
                
            except Exception as e:
                print(f" Error: {e}")
        
        # Show routing statistics
        print(f"\n ROUTING SUMMARY:")
        print("-" * 30)
        ml_routes = sum(1 for r in results if r['routing'] == 'ML')
        llm_routes = sum(1 for r in results if r['routing'] == 'LLM')
        
        print(f"ML routes: {ml_routes}/{len(results)}")
        print(f"LLM routes: {llm_routes}/{len(results)}")
        print(f"Avg confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")
        print(f"Avg time: {sum(r['time_ms'] for r in results) / len(results):.1f}ms")
        
        # System statistics
        stats = agent.get_system_statistics()
        routing_stats = stats['routing_agent']
        
        print(f"\n SYSTEM STATISTICS:")
        print("-" * 30)
        print(f"Total decisions: {routing_stats.get('total_decisions', 0)}")
        print(f"ML routing ratio: {routing_stats.get('recent_ml_ratio', 0):.1%}")
        print(f"Avg decision time: {routing_stats.get('avg_decision_time_ms', 0):.1f}ms")
        
        return True
        
    except Exception as e:
        print(f" Intelligent routing demo failed: {e}")
        return False


def demo_user_feedback():
    """Demo user feedback and learning"""
    print("\n" + "="*50)
    print(" DEMO: User Feedback & Learning")
    print("="*50)
    
    try:
        from document_classifier import MLDocumentClassifier
        from llm_classifier import LLMDocumentClassifier
        from agent import SupervisoryAgent
        from base_classifier import DocumentMetadata
        
        # Initialize system (assuming from previous demo)
        ml_classifier = MLDocumentClassifier()
        llm_classifier = LLMDocumentClassifier()
        agent = SupervisoryAgent(ml_classifier, llm_classifier)
        
        # Simulate user correction
        print("Simulating user correction scenario...")
        
        document = "Research methodology for machine learning applications in document processing"
        metadata = DocumentMetadata(file_extension="pdf")
        
        # Initial classification
        result = agent.classify_document(document, metadata)
        print(f"Initial prediction: {result.category.value}")
        
        # User correction
        correct_category = "scientific_publication"
        agent.add_user_correction(
            content=document,
            metadata=metadata,
            predicted_category=result.category.value,
            correct_category=correct_category,
            user_id="demo_user"
        )
        
        print(f" User correction added: {result.category.value} -> {correct_category}")
        
        # Show feedback statistics
        stats = agent.get_system_statistics()
        feedback_stats = stats.get('feedback_system', {})
        
        print(f"\n FEEDBACK STATISTICS:")
        print(f"Total feedback: {feedback_stats.get('total_feedback', 0)}")
        print(f"User corrections: {feedback_stats.get('user_corrections', 0)}")
        
        # Show that system is learning
        routing_stats = stats['routing_agent']
        print(f"Feedback buffer size: {routing_stats.get('feedback_buffer_size', 0)}")
        
        print(" System is now learning from user feedback!")
        
        return True
        
    except Exception as e:
        print(f" Feedback demo failed: {e}")
        return False


def main():
    """Main demo function"""
    print("Document Classification System Demo")
    print("=" * 60)
    
    # Check requirements
    print("Checking system requirements...")
    issues = check_requirements()
    
    if issues:
        print("\n System not ready. Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n Run setup first: python setup.py")
        return
    
    print(" System requirements satisfied")
    
    # Run demos
    demos = [
        ("ML Classifier", demo_ml_classifier),
        ("LLM Classifier", demo_llm_classifier), 
        ("Intelligent Routing", demo_intelligent_routing),
        ("User Feedback & Learning", demo_user_feedback)
    ]
    
    successful_demos = 0
    
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                successful_demos += 1
            else:
                print(f"  {demo_name} demo had issues")
        except KeyboardInterrupt:
            print(f"\n Demo interrupted by user")
            break
        except Exception as e:
            print(f" {demo_name} demo failed unexpectedly: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(" DEMO SUMMARY")
    print("="*60)
    print(f"Completed demos: {successful_demos}/{len(demos)}")
    
    if successful_demos == len(demos):
        print(" All demos completed successfully!")
        print("\n Your intelligent document classification system is ready!")
        print("\n Next steps:")
        print("  - Train with your own documents")
        print("  - Add user corrections to improve accuracy")
        print("  - Monitor system performance with statistics")
        print("  - Scale up for production use")
    else:
        print("  Some demos had issues. Check the error messages above.")
        print("\n Troubleshooting:")
        print("  - Ensure GROQ_API_KEY is set correctly")
        print("  - Train ML models: python document_classifier.py")
        print("  - Check internet connection for LLM API")
        print("  - Review logs for detailed error information")


if __name__ == "__main__":
    main()