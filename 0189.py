#!/usr/bin/env python3
"""
Project 189: Modern Plagiarism Detector
========================================

A comprehensive plagiarism detection system using state-of-the-art NLP techniques.
This is the legacy version - see the modern implementation in the plagiarism_detector package.

Description:
A Plagiarism Detector checks if a given document or sentence is copied—either 
partially or fully—from another source by analyzing textual similarity. This 
modern implementation uses advanced NLP techniques including sentence transformers,
BERT embeddings, and hybrid detection methods.

Features:
- Multiple detection algorithms (TF-IDF, Sentence Transformers, Hybrid)
- Web interface with FastAPI
- Database management with SQLite
- Command-line interface
- Comprehensive testing suite
- Docker support

Usage:
    # Legacy simple usage
    python 0189.py
    
    # Modern usage
    plagiarism-detector serve
    plagiarism-detector check "Your text here"
"""

# Legacy implementation for backward compatibility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def legacy_plagiarism_detection():
    """Legacy plagiarism detection implementation"""
    
    # Sample documents (could be from a database or file system)
    documents = [
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.",
        "Artificial intelligence involves training machines to perform tasks that typically require human intelligence.",
        "Machine learning allows computers to learn from data using statistical techniques.",
        "This document contains entirely original content and is not copied."
    ]
    
    # New submission to check for plagiarism
    submission = "Machine learning uses statistical methods to enable machines to learn from data."
    
    # Combine the submission with existing documents
    all_docs = [submission] + documents
    
    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Threshold for flagging plagiarism (e.g., > 0.7 = potentially plagiarized)
    threshold = 0.7
    
    print("🧠 Legacy Plagiarism Detection Results:\n")
    for idx, score in enumerate(similarity_scores):
        verdict = "⚠️ Possible Plagiarism" if score > threshold else "✅ Likely Original"
        print(f"Compared to Document {idx+1} → Similarity Score: {score:.2f} → {verdict}")

def modern_plagiarism_detection():
    """Modern plagiarism detection using the new package"""
    try:
        from plagiarism_detector import PlagiarismDetector, create_sample_documents
        
        print("🚀 Modern Plagiarism Detection Results:\n")
        
        # Initialize detector
        detector = PlagiarismDetector()
        
        # Get sample documents
        sample_docs = create_sample_documents()
        reference_documents = [doc.content for doc in sample_docs]
        
        # Test submission
        test_submission = "Machine learning uses statistical methods to enable machines to learn from data without explicit programming."
        
        # Perform hybrid detection
        results = detector.detect_plagiarism_hybrid(
            test_submission, reference_documents, sample_docs
        )
        
        print("📊 Detection Method: Hybrid (Sentence Transformer + TF-IDF)")
        print("=" * 60)
        
        max_similarity = max(result.similarity_score for result in results)
        overall_verdict = next(result.verdict for result in results if result.similarity_score == max_similarity)
        
        print(f"Overall Verdict: {overall_verdict}")
        print(f"Max Similarity: {max_similarity:.3f}")
        print()
        
        for idx, result in enumerate(results):
            print(f"Document {idx+1}: {sample_docs[idx].title}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Verdict: {result.verdict}")
            if result.matched_sentences:
                print(f"  Matched sentences: {len(result.matched_sentences)}")
            print()
            
    except ImportError:
        print("❌ Modern plagiarism detector package not installed.")
        print("   Install with: pip install -e .")
        print("   Then run: plagiarism-detector serve")
        print()

if __name__ == "__main__":
    print("=" * 80)
    print("🧠 PLAGIARISM DETECTION SYSTEM")
    print("=" * 80)
    print()
    
    # Run legacy implementation
    legacy_plagiarism_detection()
    print()
    
    # Try modern implementation
    modern_plagiarism_detection()
    
    print("=" * 80)
    print("📚 What This Project Demonstrates:")
    print("=" * 80)
    print("✅ Uses TF-IDF to represent documents numerically")
    print("✅ Computes cosine similarity to assess textual overlap")
    print("✅ Flags submissions that exceed a similarity threshold")
    print("✅ Modern implementation with advanced NLP techniques")
    print("✅ Web interface, database management, and comprehensive testing")
    print()
    print("🚀 For the full modern experience, run:")
    print("   plagiarism-detector serve")
    print("   Then visit: http://localhost:8000")
    print("=" * 80)