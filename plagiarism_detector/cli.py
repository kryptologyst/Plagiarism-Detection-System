"""
Command Line Interface for Plagiarism Detection System
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import logging

from .detector import PlagiarismDetector, create_sample_documents
from .database import DocumentDatabase

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Modern Plagiarism Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single text for plagiarism
  plagiarism-detector check "Your text here"
  
  # Check a file for plagiarism
  plagiarism-detector check --file document.txt
  
  # Add documents to the database
  plagiarism-detector add-doc --title "My Document" --content "Document content"
  
  # List all documents
  plagiarism-detector list-docs
  
  # Start web server
  plagiarism-detector serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check plagiarism command
    check_parser = subparsers.add_parser('check', help='Check text for plagiarism')
    check_parser.add_argument('text', nargs='?', help='Text to check')
    check_parser.add_argument('--file', '-f', help='File containing text to check')
    check_parser.add_argument('--method', '-m', choices=['sentence_transformer', 'tfidf', 'hybrid'], 
                            default='hybrid', help='Detection method')
    check_parser.add_argument('--output', '-o', help='Output file for results')
    check_parser.add_argument('--format', choices=['json', 'text'], default='text', 
                            help='Output format')
    
    # Add document command
    add_doc_parser = subparsers.add_parser('add-doc', help='Add document to database')
    add_doc_parser.add_argument('--title', required=True, help='Document title')
    add_doc_parser.add_argument('--content', help='Document content')
    add_doc_parser.add_argument('--file', '-f', help='File containing document content')
    add_doc_parser.add_argument('--author', help='Document author')
    add_doc_parser.add_argument('--source', help='Document source')
    
    # List documents command
    list_docs_parser = subparsers.add_parser('list-docs', help='List all documents')
    list_docs_parser.add_argument('--format', choices=['table', 'json'], default='table',
                                help='Output format')
    
    # Search documents command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize database with sample data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'check':
            check_plagiarism(args)
        elif args.command == 'add-doc':
            add_document(args)
        elif args.command == 'list-docs':
            list_documents(args)
        elif args.command == 'search':
            search_documents(args)
        elif args.command == 'stats':
            show_statistics()
        elif args.command == 'serve':
            start_server(args)
        elif args.command == 'init':
            initialize_database()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def check_plagiarism(args):
    """Check text for plagiarism"""
    # Get text to check
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Please provide text or file to check")
        return
    
    # Initialize detector and database
    detector = PlagiarismDetector()
    database = DocumentDatabase()
    
    # Get reference documents
    reference_docs = database.get_all_documents()
    if not reference_docs:
        print("Warning: No reference documents found. Using sample documents.")
        sample_docs = create_sample_documents()
        for doc in sample_docs:
            database.add_document(doc)
        reference_docs = database.get_all_documents()
    
    reference_documents = [doc.content for doc in reference_docs]
    
    # Perform detection
    if args.method == "sentence_transformer":
        results = detector.detect_plagiarism_sentence_transformer(
            text, reference_documents, reference_docs
        )
    elif args.method == "tfidf":
        results = detector.detect_plagiarism_tfidf(
            text, reference_documents, reference_docs
        )
    else:  # hybrid
        results = detector.detect_plagiarism_hybrid(
            text, reference_documents, reference_docs
        )
    
    # Save submission and results
    submission_id = database.add_submission(text)
    database.save_detection_results(submission_id, results)
    
    # Format output
    if args.format == 'json':
        output = {
            'submission_id': submission_id,
            'method': args.method,
            'results': [
                {
                    'similarity_score': result.similarity_score,
                    'confidence': result.confidence,
                    'verdict': result.verdict,
                    'matched_sentences': result.matched_sentences,
                    'source_document_id': result.source_document_id,
                    'source_document_title': result.source_document_title,
                    'detection_method': result.detection_method
                }
                for result in results
            ]
        }
        output_str = json.dumps(output, indent=2)
    else:
        # Text format
        output_lines = [
            "🧠 Plagiarism Detection Results",
            "=" * 50,
            f"Method: {args.method}",
            f"Submission ID: {submission_id}",
            ""
        ]
        
        max_similarity = max(result.similarity_score for result in results)
        overall_verdict = next(result.verdict for result in results if result.similarity_score == max_similarity)
        
        output_lines.extend([
            f"Overall Verdict: {overall_verdict}",
            f"Max Similarity: {max_similarity:.3f}",
            ""
        ])
        
        for idx, result in enumerate(results):
            output_lines.extend([
                f"Document {idx+1}: {result.source_document_title or 'Unknown'}",
                f"  Similarity Score: {result.similarity_score:.3f}",
                f"  Confidence: {result.confidence:.3f}",
                f"  Verdict: {result.verdict}",
                f"  Matched Sentences: {len(result.matched_sentences)}",
                ""
            ])
        
        output_str = "\n".join(output_lines)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"Results saved to {args.output}")
    else:
        print(output_str)


def add_document(args):
    """Add document to database"""
    # Get content
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        print("Error: Please provide content or file")
        return
    
    # Add to database
    database = DocumentDatabase()
    from .detector import DocumentMetadata
    
    doc = DocumentMetadata(
        id=f"doc_{len(database.get_all_documents()) + 1:03d}",
        title=args.title,
        content=content,
        author=args.author,
        source=args.source
    )
    
    doc_id = database.add_document(doc)
    print(f"Document added successfully with ID: {doc_id}")


def list_documents(args):
    """List all documents"""
    database = DocumentDatabase()
    documents = database.get_all_documents()
    
    if args.format == 'json':
        output = [
            {
                'id': doc.id,
                'title': doc.title,
                'author': doc.author,
                'source': doc.source,
                'word_count': doc.word_count,
                'readability_score': doc.readability_score,
                'created_at': doc.created_at.isoformat()
            }
            for doc in documents
        ]
        print(json.dumps(output, indent=2))
    else:
        if not documents:
            print("No documents found.")
            return
        
        print(f"Found {len(documents)} documents:")
        print("-" * 80)
        print(f"{'ID':<10} {'Title':<30} {'Author':<20} {'Words':<8} {'Created':<12}")
        print("-" * 80)
        
        for doc in documents:
            print(f"{doc.id:<10} {doc.title[:29]:<30} {doc.author or 'Unknown':<20} "
                  f"{doc.word_count:<8} {doc.created_at.strftime('%Y-%m-%d'):<12}")


def search_documents(args):
    """Search documents"""
    database = DocumentDatabase()
    documents = database.search_documents(args.query, args.limit)
    
    if not documents:
        print(f"No documents found matching '{args.query}'")
        return
    
    print(f"Found {len(documents)} documents matching '{args.query}':")
    print("-" * 80)
    
    for doc in documents:
        print(f"ID: {doc.id}")
        print(f"Title: {doc.title}")
        print(f"Author: {doc.author or 'Unknown'}")
        print(f"Source: {doc.source or 'Unknown'}")
        print(f"Word Count: {doc.word_count}")
        print(f"Created: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)


def show_statistics():
    """Show system statistics"""
    database = DocumentDatabase()
    stats = database.get_statistics()
    
    print("📊 System Statistics")
    print("=" * 30)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Submissions: {stats['total_submissions']}")
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Average Similarity: {stats['average_similarity']:.3f}")
    print(f"Plagiarism Rate: {stats['plagiarism_rate']:.1f}%")
    print(f"Plagiarism Cases: {stats['plagiarism_count']}")


def start_server(args):
    """Start web server"""
    import uvicorn
    from .api import app
    
    print(f"Starting web server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def initialize_database():
    """Initialize database with sample data"""
    database = DocumentDatabase()
    database.populate_sample_data()
    
    stats = database.get_statistics()
    print(f"Database initialized with {stats['total_documents']} sample documents")


if __name__ == "__main__":
    main()
