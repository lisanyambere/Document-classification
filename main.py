#!/usr/bin/env python3
"""
Main interface for document classification system
Handles document input, extraction, and intelligent classification
"""

import os
import sys
import time
import argparse
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Setup logging 
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

# Document processing
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from src.classifiers import MLDocumentClassifier, LLMDocumentClassifier
from src.agents import SupervisoryAgent
from src.core import DocumentMetadata, ClassificationResult


class DocumentExtractor:
    
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx', '.doc', '.md'}
        if not HAS_PDF:
            print("Warning: PyPDF2 not installed. PDF support disabled.")
        if not HAS_DOCX:
            print("Warning: python-docx not installed. DOCX support disabled.")
    
    # Extract content and metadata from document file
    def extract_document(self, file_path: str) -> tuple[str, DocumentMetadata]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        
        # Extract content based on file type
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            content = self._extract_pdf(file_path)
        elif extension in {'.docx', '.doc'}:
            content = self._extract_docx(file_path)
        elif extension in {'.txt', '.md'}:
            content = self._extract_text(file_path)
        else:
            # Try as plain text
            try:
                content = self._extract_text(file_path)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {extension}. Error: {e}")
        
        return content, metadata
    
    # Extract file metadata
    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        stat = file_path.stat()
        
        # Try to detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return DocumentMetadata(
            file_extension=file_path.suffix.lower().lstrip('.'),
            file_size=stat.st_size,
            sender_email=None  # Can't extract from file system
        )
    
    # Extract text from PDF
    def _extract_pdf(self, file_path: Path) -> str:
        if not HAS_PDF:
            raise ImportError("PyPDF2 required for PDF processing. Install: pip install PyPDF2")
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {e}")
    
    # Extract text from DOCX
    def _extract_docx(self, file_path: Path) -> str:
        if not HAS_DOCX:
            raise ImportError("python-docx required for DOCX processing. Install: pip install python-docx")
        
        try:
            doc = DocxDocument(str(file_path))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract DOCX content: {e}")
    
    # Extract text from plain text file
    def _extract_text(self, file_path: Path) -> str:
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read().strip()
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode file with any supported encoding")
        except Exception as e:
            raise ValueError(f"Failed to extract text content: {e}")


class DocumentClassificationInterface:
    
    # Initialize the classification system
    def __init__(self):
        print("Initializing document classification system...")
        
        # Initialize extractors and classifiers
        self.extractor = DocumentExtractor()
        
        try:
            self.ml_classifier = MLDocumentClassifier()
            if not self.ml_classifier.is_ready():
                print("Warning: ML classifier not ready. Run: python document_classifier.py")
                
        except Exception as e:
            print(f"Error initializing ML classifier: {e}")
            self.ml_classifier = None
        
        try:
            self.llm_classifier = LLMDocumentClassifier()
            if not self.llm_classifier.is_ready():
                print("Warning: LLM classifier not ready. Check GROQ_API_KEY in .env")
        except Exception as e:
            print(f"Error initializing LLM classifier: {e}")
            self.llm_classifier = None
        
        # Initialize agent if both classifiers available
        if self.ml_classifier and self.llm_classifier:
            self.agent = SupervisoryAgent(self.ml_classifier, self.llm_classifier)
            print("Enhanced agent ready!")
        else:
            self.agent = None
            print("Warning: Agent unavailable. Need both ML and LLM classifiers.")
    
    # Classify a single document
    def classify_document(self, file_path: str, output_format: str = 'text') -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Extract document content and metadata
            print(f"Processing: {file_path}")
            content, metadata = self.extractor.extract_document(file_path)
            
            print(f"Extracted {len(content)} characters from {metadata.file_extension} file")
            
            # Classify using agent or fallback
            if self.agent:
                result = self.agent.classify_document(content, metadata)
                routing_info = result.features_used or {}
                routing_decision = routing_info.get('routing_decision', 'Unknown')
                predicted_confidence = routing_info.get('predicted_ml_confidence', 0)
            elif self.llm_classifier:
                result = self.llm_classifier.predict(content, metadata)
                routing_decision = 'LLM'
                predicted_confidence = 0
            elif self.ml_classifier:
                result = self.ml_classifier.predict(content, metadata)
                routing_decision = 'ML'
                predicted_confidence = result.confidence
            else:
                raise RuntimeError("No classifiers available")
            
            total_time = (time.time() - start_time) * 1000
            
            # Prepare result
            classification_result = {
                'file_path': file_path,
                'category': result.category.value,
                'confidence': result.confidence,
                'classifier_used': routing_decision,
                'processing_time_ms': total_time,
                'content_length': len(content),
                'file_size': metadata.file_size,
                'file_extension': metadata.file_extension
            }
            
            # Add detailed info if requested
            if output_format == 'detailed':
                full_reasoning = getattr(result, 'reasoning', None)
                classification_result.update({
                    'predicted_ml_confidence': predicted_confidence,
                    'reasoning': full_reasoning,
                    'content_preview': content[:200] + "..." if len(content) > 200 else content,
                    'metadata': metadata.to_dict() if hasattr(metadata, 'to_dict') else str(metadata)
                })
                
                # Log full reasoning to console for visibility
                if full_reasoning:
                    print(f"FULL REASONING for {Path(file_path).name}:")
                    print(f"Classifier: {routing_decision}")
                    print(f"Reasoning: {full_reasoning}")
                    print("-" * 60)
            
            return classification_result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    # Classify multiple documents
    def classify_batch(self, file_paths: List[str], output_format: str = 'text') -> List[Dict[str, Any]]:
        print(f"Processing {len(file_paths)} documents...")
        
        results = []
        successful = 0
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")
            
            result = self.classify_document(file_path, output_format)
            results.append(result)
            
            if 'error' not in result:
                successful += 1
                print(f"Result: {result['category']} ({result['confidence']:.3f} via {result['classifier_used']})")
            else:
                print(f"Error: {result['error']}")
        
        print(f"\nBatch complete: {successful}/{len(file_paths)} successful")
        return results
    
    # Get system performance statistics
    def get_system_stats(self) -> Dict[str, Any]:
        if not self.agent:
            return {'error': 'Agent not available'}
        
        stats = self.agent.get_system_statistics()
        
        # Simplify for display
        return {
            'total_classifications': stats['system']['total_classifications'],
            'ml_classifier_ready': self.ml_classifier.is_ready() if self.ml_classifier else False,
            'llm_classifier_ready': self.llm_classifier.is_ready() if self.llm_classifier else False,
            'routing_decisions': stats['routing_agent'].get('total_decisions', 0),
            'average_confidence': stats['ml_classifier'].get('avg_confidence', 0),
            'supported_formats': list(self.extractor.supported_formats)
        }


# Print classification result
def print_result(result: Dict[str, Any], output_format: str = 'text'):
    if 'error' in result:
        print(f"ERROR processing {result['file_path']}: {result['error']}")
        return
    
    file_name = Path(result['file_path']).name
    
    if output_format == 'json':
        print(json.dumps(result, indent=2))
    elif output_format == 'detailed':
        print(f"File: {file_name}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Classifier: {result['classifier_used']}")
        print(f"Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"Content length: {result['content_length']} chars")
        if 'reasoning' in result and result['reasoning']:
            print(f"Reasoning: {result['reasoning']}")
        if 'content_preview' in result:
            print(f"Preview: {result['content_preview']}")
    else:  # text
        print(f"{file_name} -> {result['category']} ({result['confidence']:.3f} via {result['classifier_used']})")


# Main command line interface
def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Document Classification System',
        epilog='Examples:\n'
               '  python main.py document.pdf\n'
               '  python main.py *.pdf --format json\n'
               '  python main.py folder/ --batch --format detailed\n'
               '  python main.py --stats',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('files', nargs='*', help='Document files or directories to classify')
    parser.add_argument('--batch', action='store_true', help='Process multiple files')
    parser.add_argument('--format', choices=['text', 'json', 'detailed'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize system
    classifier = DocumentClassificationInterface()
    
    # Show stats if requested
    if args.stats:
        stats = classifier.get_system_stats()
        print("\nSystem Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    # Check if files provided
    if not args.files:
        print("No files specified. Use --help for usage information.")
        return
    
    # Collect file paths
    file_paths = []
    for file_arg in args.files:
        path = Path(file_arg)
        if path.is_file():
            file_paths.append(str(path))
        elif path.is_dir():
            # Find supported files in directory
            for ext in classifier.extractor.supported_formats:
                file_paths.extend(str(p) for p in path.glob(f"*{ext}"))
        elif '*' in file_arg:
            # Handle wildcards
            import glob
            file_paths.extend(glob.glob(file_arg))
        else:
            print(f"Warning: {file_arg} not found")
    
    if not file_paths:
        print("No valid files found to process.")
        return
    
    # Process files
    if len(file_paths) == 1 and not args.batch:
        # Single file
        result = classifier.classify_document(file_paths[0], args.format)
        print_result(result, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    else:
        # Multiple files
        results = classifier.classify_batch(file_paths, args.format)
        
        print(f"\n{'='*60}")
        print("BATCH RESULTS")
        print('='*60)
        
        for result in results:
            print_result(result, args.format)
        
        # Summary
        successful = [r for r in results if 'error' not in r]
        if successful:
            categories = {}
            total_time = sum(r['processing_time_ms'] for r in successful)
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            
            for result in successful:
                cat = result['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\nSUMMARY:")
            print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results):.1%})")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Total processing time: {total_time:.1f}ms")
            print(f"Categories found: {dict(sorted(categories.items()))}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()