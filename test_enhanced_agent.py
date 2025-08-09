#!/usr/bin/env python3
"""
Test script for enhanced agent with LLM-based routing
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

def test_enhanced_agent():
    """Test the enhanced agent with LLM reasoning"""
    print("=" * 60)
    print("Enhanced Agent Test - LLM Routing")
    print("=" * 60)
    
    try:
        # Import classifiers and agent
        from document_classifier import MLDocumentClassifier
        from llm_classifier import LLMDocumentClassifier
        from agent import SupervisoryAgent
        from base_classifier import DocumentMetadata
        
        # Initialize classifiers
        print("1. Initializing ML classifier...")
        ml_classifier = MLDocumentClassifier()
        if not ml_classifier.is_ready():
            print("ERROR: ML classifier not ready (run: python document_classifier.py)")
            return False
        print("   ML classifier ready!")
        
        print("2. Initializing LLM classifier...")
        llm_classifier = LLMDocumentClassifier()
        if not llm_classifier.is_ready():
            print("ERROR: LLM classifier not ready (check GROQ_API_KEY)")
            return False
        print("   LLM classifier ready!")
        
        print("3. Initializing enhanced agent...")
        agent = SupervisoryAgent(ml_classifier, llm_classifier)
        print("   Enhanced agent ready!")
        
        # Test documents with different complexity levels
        test_documents = [
            {
                "name": "Simple Invoice",
                "content": "Invoice #001 for consulting services. Amount: $1,500. Due: 30 days.",
                "metadata": DocumentMetadata(file_extension="pdf"),
                "expected_route": "ML"  # Simple, should use ML
            },
            {
                "name": "Complex Technical Specification",
                "content": """
                TECHNICAL SPECIFICATION DOCUMENT v2.1
                
                System Architecture for Advanced Classification Pipeline
                
                1. METHODOLOGY OVERVIEW
                This specification outlines the implementation methodology for a multi-tier 
                document classification system utilizing both traditional machine learning 
                algorithms and large language model architectures.
                
                2. ALGORITHM SPECIFICATIONS  
                - Primary classifier: Gradient Boosting with feature engineering
                - Secondary classifier: Transformer-based LLM with attention mechanisms
                - Routing algorithm: Adaptive threshold with confidence prediction
                - Performance metrics: Accuracy >95%, Latency <200ms, Cost <$0.01/doc
                
                3. TECHNICAL IMPLEMENTATION
                The system employs a hybrid approach combining the speed of traditional ML
                with the contextual understanding of modern LLMs. Document features are
                extracted using TF-IDF vectorization enhanced with semantic embeddings.
                """,
                "metadata": DocumentMetadata(
                    sender_email="architect@company.com",
                    file_extension="pdf",
                    file_size=4096
                ),
                "expected_route": "LLM"  # Complex technical content, should use LLM
            },
            {
                "name": "Business Letter",
                "content": "Dear Mr. Johnson, Thank you for your inquiry about our services. We would be happy to discuss your project requirements in detail.",
                "metadata": DocumentMetadata(file_extension="txt"),
                "expected_route": "ML"  # Simple business content, should use ML
            },
            {
                "name": "Research Report Abstract",
                "content": """
                Abstract: A Comparative Analysis of Document Classification Methodologies
                
                This study presents a comprehensive evaluation of traditional machine learning
                approaches versus modern transformer-based architectures for document 
                classification tasks. Our methodology incorporates both quantitative performance 
                metrics and qualitative analysis of classification accuracy across diverse 
                document categories including technical specifications, business correspondence,
                and regulatory compliance documentation.
                """,
                "metadata": DocumentMetadata(
                    sender_email="researcher@university.edu", 
                    file_extension="pdf"
                ),
                "expected_route": "LLM"  # Academic content with mixed terminology
            }
        ]
        
        print("\n4. Testing enhanced routing decisions...")
        print("-" * 40)
        
        results = []
        total_start_time = time.time()
        
        for i, doc in enumerate(test_documents, 1):
            print(f"\n[{i}] Testing: {doc['name']}")
            print(f"Content preview: {doc['content'][:80]}...")
            
            start_time = time.time()
            try:
                # Classify using enhanced agent
                result = agent.classify_document(doc['content'], doc['metadata'])
                processing_time = (time.time() - start_time) * 1000
                
                # Extract routing information
                routing_info = result.features_used or {}
                routing_decision = routing_info.get('routing_decision', 'Unknown')
                predicted_confidence = routing_info.get('predicted_ml_confidence', 0)
                
                print(f"Result: {result.category.value}")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Routed to: {routing_decision} classifier")
                print(f"Predicted ML confidence: {predicted_confidence:.3f}")
                print(f"Processing time: {processing_time:.1f}ms")
                
                # Check if routing matches expectation
                routing_correct = routing_decision == doc['expected_route']
                routing_status = "CORRECT" if routing_correct else "UNEXPECTED"
                print(f"Routing: {routing_status} (expected {doc['expected_route']})")
                
                results.append({
                    'name': doc['name'],
                    'category': result.category.value,
                    'confidence': result.confidence,
                    'routing': routing_decision,
                    'expected_routing': doc['expected_route'],
                    'routing_correct': routing_correct,
                    'processing_time_ms': processing_time,
                    'classifier_type': result.classifier_type
                })
                
            except Exception as e:
                print(f"ERROR: Classification failed - {e}")
                results.append({
                    'name': doc['name'],
                    'error': str(e)
                })
        
        total_time = (time.time() - total_start_time) * 1000
        
        # Analysis and summary
        print("\n" + "=" * 60)
        print("ENHANCED ROUTING ANALYSIS")
        print("=" * 60)
        
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            # Routing analysis
            ml_routes = sum(1 for r in successful_results if r['routing'] == 'ML')
            llm_routes = sum(1 for r in successful_results if r['routing'] == 'LLM')
            routing_accuracy = sum(1 for r in successful_results if r['routing_correct']) / len(successful_results)
            
            print(f"Total classifications: {len(successful_results)}")
            print(f"ML routes: {ml_routes}")
            print(f"LLM routes: {llm_routes}")
            print(f"Routing accuracy: {routing_accuracy:.1%}")
            print(f"Average confidence: {sum(r['confidence'] for r in successful_results) / len(successful_results):.3f}")
            print(f"Average processing time: {sum(r['processing_time_ms'] for r in successful_results) / len(successful_results):.1f}ms")
            print(f"Total time: {total_time:.1f}ms")
            
            # System statistics
            stats = agent.get_system_statistics()
            routing_stats = stats['routing_agent']
            
            print(f"\nSYSTEM STATISTICS:")
            print(f"Total routing decisions: {routing_stats.get('total_decisions', 0)}")
            print(f"LLM reasoning decisions: {agent.routing_agent.llm_routing_decisions}")
            print(f"Heuristic decisions: {agent.routing_agent.heuristic_routing_decisions}")
            
            if agent.routing_agent.llm_routing_decisions > 0:
                llm_ratio = agent.routing_agent.llm_routing_decisions / (agent.routing_agent.llm_routing_decisions + agent.routing_agent.heuristic_routing_decisions)
                print(f"LLM reasoning ratio: {llm_ratio:.1%}")
            
            print(f"\nCLASSIFICATION DETAILS:")
            print("-" * 30)
            for r in successful_results:
                status = "PASS" if r['routing_correct'] else "FAIL"
                print(f"{status} {r['name']}: {r['category']} via {r['routing']} ({r['confidence']:.2f})")
        
        failed_results = [r for r in results if 'error' in r]
        if failed_results:
            print(f"\nFAILED CLASSIFICATIONS: {len(failed_results)}")
            for r in failed_results:
                print(f"FAIL {r['name']}: {r['error']}")
        
        success_rate = len(successful_results) / len(results)
        print(f"\nOVERALL SUCCESS RATE: {success_rate:.1%}")
        
        if success_rate >= 0.75 and routing_accuracy >= 0.5:
            print("\nEnhanced agent test PASSED!")
            print("ML models working")
            print("LLM classifier working") 
            print("Enhanced routing with LLM reasoning operational")
            return True
        else:
            print("\nEnhanced agent test had issues")
            print(f"Success rate: {success_rate:.1%} (need >75%)")
            print(f"Routing accuracy: {routing_accuracy:.1%} (need >50%)")
            return False
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Enhanced Document Classification Agent Test")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_enhanced_agent()
    
    if success:
        print("\nSystem ready for production use!")
        print("\nFeatures working:")
        print("- ML classifier (98.4% accuracy)")
        print("- LLM classifier with GROQ API")  
        print("- Enhanced routing with LLM reasoning")
        print("- Intelligent document analysis")
        print("- Performance monitoring")
    else:
        print("\nSystem needs attention:")
        print("- Check GROQ_API_KEY in .env")
        print("- Ensure ML models are trained")
        print("- Review error messages above")

if __name__ == "__main__":
    main()