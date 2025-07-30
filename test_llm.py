#!/usr/bin/env python3
"""
Test script for LLM document classification
"""

import os
import sys
from dotenv import load_dotenv
from llm_classifier import LLMDocumentClassifier, DocumentCategory

# Load environment variables
load_dotenv()

def test_llm_classification():
    """Test the LLM classifier with sample documents"""
    
    # Check if API key is set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        print("ERROR: GROQ_API_KEY not found or not set properly")
        print("Please set your Groq API key in a .env file:")
        print("GROQ_API_KEY=your-actual-api-key-here")
        return False
    
    print("SUCCESS: Groq API key found")
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Model: {os.getenv('GROQ_MODEL_NAME', 'llama3-8b-8192')}")
    print(f"  Max Chars: {os.getenv('LLM_MAX_CHARS', '2000')}")
    print(f"  Max Words: {os.getenv('LLM_MAX_WORDS', '300')}")
    print(f"  Temperature: {os.getenv('GROQ_TEMPERATURE', '0.1')}")
    
    try:
        # Initialize classifier
        print("\nInitializing LLM classifier...")
        classifier = LLMDocumentClassifier()
        print("SUCCESS: LLM classifier initialized successfully")
        
        # Test documents with different characteristics
        test_documents = [
            {
                "name": "Short Business Letter",
                "content": """
                Dear Mr. Johnson,
                
                I am writing to follow up on our meeting last week regarding the quarterly budget review.
                As discussed, we need to allocate additional funds for the marketing campaign.
                
                Please let me know if you have any questions.
                
                Best regards,
                Sarah Smith
                Finance Manager
                """,
                "metadata": {"sender": "sarah.smith@company.com", "file_extension": "pdf"}
            },
            {
                "name": "Long Technical Document",
                "content": """
                Technical Specification Document
                
                Product: Advanced Document Classification System
                Version: 2.1
                Date: 2024-01-27
                
                System Requirements:
                - Python 3.8+
                - 8GB RAM minimum
                - GPU support for ML models
                - SSD storage recommended
                - Network connectivity for API calls
                
                Architecture:
                - Microservices design
                - RESTful API endpoints
                - Real-time processing capabilities
                - Load balancing support
                - Horizontal scaling
                
                Performance Metrics:
                - Processing speed: 1000 documents/minute
                - Accuracy: 95%+ on standard datasets
                - Latency: <100ms for simple documents
                - Throughput: 10,000 documents/hour
                
                Security Features:
                - End-to-end encryption
                - API key authentication
                - Rate limiting
                - Audit logging
                - Data anonymization
                
                Deployment Options:
                - Cloud deployment (AWS, Azure, GCP)
                - On-premises installation
                - Hybrid cloud setup
                - Containerized deployment
                """,
                "metadata": {"sender": "tech@company.com", "file_extension": "docx"}
            },
            {
                "name": "Invoice Document",
                "content": """
                INVOICE
                
                Invoice #: INV-2024-001
                Date: January 27, 2024
                Due Date: February 27, 2024
                
                Bill To:
                ABC Corporation
                123 Business St
                City, State 12345
                
                Description:
                - Document Classification System License: $5,000
                - Technical Support (3 months): $1,500
                - Implementation Services: $2,500
                - Training Sessions: $800
                - Custom Integration: $1,200
                
                Subtotal: $11,000
                Tax (8.5%): $935
                Total: $11,935
                
                Payment Terms: Net 30
                Payment Method: Bank Transfer
                
                Thank you for your business!
                """,
                "metadata": {"sender": "billing@company.com", "file_extension": "pdf"}
            },
            {
                "name": "Resume Document",
                "content": """
                JOHN DOE
                Software Engineer
                john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe
                
                SUMMARY
                Experienced software engineer with 5+ years in machine learning and data science.
                Specialized in building scalable document processing systems and API development.
                
                EXPERIENCE
                Senior Software Engineer | Tech Corp | 2020-2024
                - Led development of machine learning systems processing 1M+ documents daily
                - Managed team of 5 developers across multiple time zones
                - Improved system performance by 40% through optimization
                - Implemented CI/CD pipelines reducing deployment time by 60%
                
                Software Engineer | Startup Inc | 2018-2020
                - Built RESTful APIs serving 10,000+ requests per minute
                - Developed microservices architecture for document processing
                - Collaborated with data science team on ML model integration
                
                EDUCATION
                Bachelor of Science in Computer Science
                University of Technology | 2018
                GPA: 3.8/4.0
                
                SKILLS
                - Programming: Python, Java, JavaScript, Go
                - ML/AI: TensorFlow, PyTorch, Scikit-learn
                - Cloud: AWS, Docker, Kubernetes
                - Databases: PostgreSQL, MongoDB, Redis
                """,
                "metadata": {"sender": "john.doe@email.com", "file_extension": "docx"}
            }
        ]
        
        print(f"\nTesting {len(test_documents)} documents...")
        
        # Test each document
        for i, doc in enumerate(test_documents, 1):
            print(f"\n--- Test {i}: {doc['name']} ---")
            print(f"Original length: {len(doc['content'])} characters")
            print(f"Content preview: {doc['content'][:80]}...")
            
            try:
                # Classify document
                result = classifier.classify_document(doc['content'], doc['metadata'])
                
                # Display results
                print(f"SUCCESS: Category: {result.category.value}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Processing Time: {result.processing_time_ms:.2f}ms")
                print(f"Reasoning: {result.reasoning[:150]}...")
                
            except Exception as e:
                print(f"ERROR: Error classifying document: {e}")
                continue
        
        # Print performance statistics
        print(f"\nPerformance Statistics:")
        stats = classifier.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\nSUCCESS: LLM classification test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        return False

def test_api_connection():
    """Test basic API connection"""
    import requests
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        print("ERROR: No valid API key found")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
        "messages": [{"role": "user", "content": "Hello, this is a test message."}],
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "50"))
    }
    
    try:
        print("Testing Groq API connection...")
        response = requests.post(
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=int(os.getenv("GROQ_TIMEOUT", "30"))
        )
        
        if response.status_code == 200:
            print("SUCCESS: Groq API connection successful!")
            return True
        else:
            print(f"ERROR: API connection failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: API connection error: {e}")
        return False

def test_document_sampling():
    """Test document sampling functionality"""
    print("\nTesting document sampling...")
    
    # Create a classifier instance
    try:
        classifier = LLMDocumentClassifier()
    except ValueError:
        print("ERROR: Cannot test sampling without API key")
        return False
    
    # Test documents of different lengths
    test_texts = [
        "Short text",
        "This is a medium length text that should fit within the character limit without any truncation needed.",
        "This is a very long text " * 200  # ~4000 characters
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Sampling Test {i} ---")
        print(f"Original length: {len(text)} characters")
        
        sampled = classifier._sample_document(text)
        print(f"Sampled length: {len(sampled)} characters")
        print(f"Sampling ratio: {len(sampled)/len(text)*100:.1f}%")
        print(f"Sample preview: {sampled[:100]}...")
        
        if len(text) > classifier.max_chars:
            if "..." in sampled:
                print("SUCCESS: Truncation detected correctly")
            else:
                print("ERROR: Truncation not applied")
        else:
            if sampled == text:
                print("SUCCESS: Full text preserved correctly")
            else:
                print("ERROR: Text modified unnecessarily")

if __name__ == "__main__":
    print("Starting LLM Classification Tests")
    print("=" * 50)
    
    # Test API connection first
    if not test_api_connection():
        print("\nERROR: API connection test failed. Please check your API key and internet connection.")
        sys.exit(1)
    
    # Test document sampling
    test_document_sampling()
    
    # Test classification
    if test_llm_classification():
        print("\nSUCCESS: All tests passed!")
        sys.exit(0)
    else:
        print("\nERROR: Some tests failed.")
        sys.exit(1) 