"""
Modern Plagiarism Detection Engine
Uses state-of-the-art NLP techniques including sentence transformers and BERT embeddings
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import textstat

# Download required NLTK data
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlagiarismResult:
    """Data class for plagiarism detection results"""
    similarity_score: float
    confidence: float
    verdict: str
    matched_sentences: List[str]
    source_document_id: Optional[str] = None
    source_document_title: Optional[str] = None
    detection_method: str = "sentence_transformer"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DocumentMetadata:
    """Metadata for documents in the database"""
    id: str
    title: str
    content: str
    author: Optional[str] = None
    source: Optional[str] = None
    created_at: datetime = None
    word_count: int = 0
    readability_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.readability_score == 0.0:
            self.readability_score = textstat.flesch_reading_ease(self.content)


class TextPreprocessor:
    """Advanced text preprocessing for better plagiarism detection"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        return sent_tokenize(text)


class PlagiarismDetector:
    """Modern plagiarism detection engine using multiple techniques"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the plagiarism detector
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        
        # Initialize models
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Detection thresholds
        self.thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
    
    def detect_plagiarism_sentence_transformer(
        self, 
        submission: str, 
        reference_documents: List[str],
        document_metadata: Optional[List[DocumentMetadata]] = None
    ) -> List[PlagiarismResult]:
        """
        Detect plagiarism using sentence transformer embeddings
        
        Args:
            submission: Text to check for plagiarism
            reference_documents: List of reference documents
            document_metadata: Optional metadata for reference documents
            
        Returns:
            List of PlagiarismResult objects
        """
        results = []
        
        # Preprocess submission
        submission_sentences = self.preprocessor.extract_sentences(submission)
        submission_embedding = self.sentence_model.encode([submission])
        
        for idx, ref_doc in enumerate(reference_documents):
            # Preprocess reference document
            ref_sentences = self.preprocessor.extract_sentences(ref_doc)
            ref_embedding = self.sentence_model.encode([ref_doc])
            
            # Calculate similarity
            similarity = cosine_similarity(submission_embedding, ref_embedding)[0][0]
            
            # Find matching sentences
            matched_sentences = self._find_matching_sentences(
                submission_sentences, ref_sentences
            )
            
            # Determine confidence and verdict
            confidence, verdict = self._determine_verdict(similarity)
            
            # Create result
            result = PlagiarismResult(
                similarity_score=similarity,
                confidence=confidence,
                verdict=verdict,
                matched_sentences=matched_sentences,
                source_document_id=document_metadata[idx].id if document_metadata else None,
                source_document_title=document_metadata[idx].title if document_metadata else None,
                detection_method="sentence_transformer"
            )
            
            results.append(result)
        
        return results
    
    def detect_plagiarism_tfidf(
        self, 
        submission: str, 
        reference_documents: List[str],
        document_metadata: Optional[List[DocumentMetadata]] = None
    ) -> List[PlagiarismResult]:
        """
        Detect plagiarism using TF-IDF vectorization (fallback method)
        
        Args:
            submission: Text to check for plagiarism
            reference_documents: List of reference documents
            document_metadata: Optional metadata for reference documents
            
        Returns:
            List of PlagiarismResult objects
        """
        results = []
        
        # Combine all documents for vectorization
        all_docs = [submission] + reference_documents
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs)
        
        # Calculate similarities
        submission_vector = tfidf_matrix[0:1]
        reference_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(submission_vector, reference_vectors)[0]
        
        for idx, similarity in enumerate(similarities):
            # Determine confidence and verdict
            confidence, verdict = self._determine_verdict(similarity)
            
            # Create result
            result = PlagiarismResult(
                similarity_score=similarity,
                confidence=confidence,
                verdict=verdict,
                matched_sentences=[],  # TF-IDF doesn't provide sentence-level matches
                source_document_id=document_metadata[idx].id if document_metadata else None,
                source_document_title=document_metadata[idx].title if document_metadata else None,
                detection_method="tfidf"
            )
            
            results.append(result)
        
        return results
    
    def detect_plagiarism_hybrid(
        self, 
        submission: str, 
        reference_documents: List[str],
        document_metadata: Optional[List[DocumentMetadata]] = None
    ) -> List[PlagiarismResult]:
        """
        Hybrid plagiarism detection combining multiple methods
        
        Args:
            submission: Text to check for plagiarism
            reference_documents: List of reference documents
            document_metadata: Optional metadata for reference documents
            
        Returns:
            List of PlagiarismResult objects
        """
        # Get results from both methods
        st_results = self.detect_plagiarism_sentence_transformer(
            submission, reference_documents, document_metadata
        )
        tfidf_results = self.detect_plagiarism_tfidf(
            submission, reference_documents, document_metadata
        )
        
        # Combine results using weighted average
        hybrid_results = []
        for st_result, tfidf_result in zip(st_results, tfidf_results):
            # Weighted combination (sentence transformer gets higher weight)
            combined_score = 0.7 * st_result.similarity_score + 0.3 * tfidf_result.similarity_score
            combined_confidence = 0.7 * st_result.confidence + 0.3 * tfidf_result.confidence
            
            # Use sentence transformer's matched sentences
            confidence, verdict = self._determine_verdict(combined_score)
            
            hybrid_result = PlagiarismResult(
                similarity_score=combined_score,
                confidence=combined_confidence,
                verdict=verdict,
                matched_sentences=st_result.matched_sentences,
                source_document_id=st_result.source_document_id,
                source_document_title=st_result.source_document_title,
                detection_method="hybrid"
            )
            
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def _find_matching_sentences(
        self, 
        submission_sentences: List[str], 
        reference_sentences: List[str],
        threshold: float = 0.7
    ) -> List[str]:
        """Find sentences that are similar between submission and reference"""
        matched_sentences = []
        
        if not submission_sentences or not reference_sentences:
            return matched_sentences
        
        # Encode all sentences
        all_sentences = submission_sentences + reference_sentences
        sentence_embeddings = self.sentence_model.encode(all_sentences)
        
        submission_embeddings = sentence_embeddings[:len(submission_sentences)]
        reference_embeddings = sentence_embeddings[len(submission_sentences):]
        
        # Find similar sentences
        similarities = cosine_similarity(submission_embeddings, reference_embeddings)
        
        for i, submission_sent in enumerate(submission_sentences):
            max_similarity = np.max(similarities[i])
            if max_similarity > threshold:
                best_match_idx = np.argmax(similarities[i])
                matched_sentences.append(reference_sentences[best_match_idx])
        
        return matched_sentences
    
    def _determine_verdict(self, similarity_score: float) -> Tuple[float, str]:
        """Determine plagiarism verdict based on similarity score"""
        if similarity_score >= self.thresholds['high_confidence']:
            return 0.9, "🚨 High Probability of Plagiarism"
        elif similarity_score >= self.thresholds['medium_confidence']:
            return 0.7, "⚠️ Possible Plagiarism"
        elif similarity_score >= self.thresholds['low_confidence']:
            return 0.5, "🔍 Low Similarity Detected"
        else:
            return 0.1, "✅ Likely Original"
    
    def batch_detect(
        self, 
        submissions: List[str], 
        reference_documents: List[str],
        document_metadata: Optional[List[DocumentMetadata]] = None,
        method: str = "hybrid"
    ) -> Dict[str, List[PlagiarismResult]]:
        """
        Batch plagiarism detection for multiple submissions
        
        Args:
            submissions: List of texts to check
            reference_documents: List of reference documents
            document_metadata: Optional metadata for reference documents
            method: Detection method ('sentence_transformer', 'tfidf', 'hybrid')
            
        Returns:
            Dictionary mapping submission indices to results
        """
        results = {}
        
        for idx, submission in enumerate(submissions):
            if method == "sentence_transformer":
                submission_results = self.detect_plagiarism_sentence_transformer(
                    submission, reference_documents, document_metadata
                )
            elif method == "tfidf":
                submission_results = self.detect_plagiarism_tfidf(
                    submission, reference_documents, document_metadata
                )
            else:  # hybrid
                submission_results = self.detect_plagiarism_hybrid(
                    submission, reference_documents, document_metadata
                )
            
            results[str(idx)] = submission_results
        
        return results


def create_sample_documents() -> List[DocumentMetadata]:
    """Create sample documents for testing"""
    sample_docs = [
        DocumentMetadata(
            id="doc_001",
            title="Introduction to Machine Learning",
            content="Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data without being explicitly programmed.",
            author="Dr. Smith",
            source="Academic Journal"
        ),
        DocumentMetadata(
            id="doc_002", 
            title="AI Fundamentals",
            content="Artificial intelligence involves training machines to perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
            author="Prof. Johnson",
            source="Textbook"
        ),
        DocumentMetadata(
            id="doc_003",
            title="Data Science Overview", 
            content="Data science combines statistical analysis, machine learning, and domain expertise to extract insights from structured and unstructured data.",
            author="Dr. Brown",
            source="Research Paper"
        ),
        DocumentMetadata(
            id="doc_004",
            title="Original Research",
            content="This document contains entirely original content and is not copied from any other source. It represents novel research findings.",
            author="Dr. Wilson",
            source="Original Research"
        )
    ]
    return sample_docs


if __name__ == "__main__":
    # Example usage
    detector = PlagiarismDetector()
    
    # Sample documents
    sample_docs = create_sample_documents()
    reference_documents = [doc.content for doc in sample_docs]
    
    # Test submission
    test_submission = "Machine learning uses statistical methods to enable machines to learn from data without explicit programming."
    
    print("🧠 Modern Plagiarism Detection Results:\n")
    
    # Test different methods
    methods = ["sentence_transformer", "tfidf", "hybrid"]
    
    for method in methods:
        print(f"\n📊 Method: {method.upper()}")
        print("-" * 50)
        
        results = detector.detect_plagiarism_hybrid(
            test_submission, reference_documents, sample_docs
        )
        
        for idx, result in enumerate(results):
            print(f"Document {idx+1}: {sample_docs[idx].title}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Verdict: {result.verdict}")
            if result.matched_sentences:
                print(f"  Matched sentences: {len(result.matched_sentences)}")
            print()
