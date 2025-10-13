"""
Comprehensive test suite for plagiarism detection system
"""

import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from plagiarism_detector.detector import (
    PlagiarismDetector, 
    DocumentMetadata, 
    PlagiarismResult,
    TextPreprocessor,
    create_sample_documents
)
from plagiarism_detector.database import DocumentDatabase


class TestTextPreprocessor:
    """Test text preprocessing functionality"""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_preprocess_basic(self):
        """Test basic text preprocessing"""
        text = "This is a TEST sentence with some STOP words."
        result = self.preprocessor.preprocess(text)
        
        # Should be lowercase and stopwords removed
        assert "test" in result
        assert "sentence" in result
        assert "stop" not in result
        assert "this" not in result  # stopword
    
    def test_extract_sentences(self):
        """Test sentence extraction"""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.preprocessor.extract_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence." in sentences
        assert "Second sentence!" in sentences
        assert "Third sentence?" in sentences


class TestDocumentMetadata:
    """Test DocumentMetadata class"""
    
    def test_document_creation(self):
        """Test document metadata creation"""
        doc = DocumentMetadata(
            id="test_001",
            title="Test Document",
            content="This is a test document with some content.",
            author="Test Author",
            source="Test Source"
        )
        
        assert doc.id == "test_001"
        assert doc.title == "Test Document"
        assert doc.content == "This is a test document with some content."
        assert doc.author == "Test Author"
        assert doc.source == "Test Source"
        assert doc.word_count > 0
        assert doc.readability_score > 0
        assert isinstance(doc.created_at, datetime)
    
    def test_auto_calculations(self):
        """Test automatic calculations"""
        content = "This is a test document with exactly ten words here."
        doc = DocumentMetadata(
            id="test_002",
            title="Test",
            content=content
        )
        
        assert doc.word_count == 10
        assert doc.readability_score > 0


class TestPlagiarismResult:
    """Test PlagiarismResult class"""
    
    def test_result_creation(self):
        """Test plagiarism result creation"""
        result = PlagiarismResult(
            similarity_score=0.85,
            confidence=0.9,
            verdict="High plagiarism",
            matched_sentences=["Sentence 1", "Sentence 2"],
            source_document_id="doc_001",
            source_document_title="Test Document"
        )
        
        assert result.similarity_score == 0.85
        assert result.confidence == 0.9
        assert result.verdict == "High plagiarism"
        assert len(result.matched_sentences) == 2
        assert result.source_document_id == "doc_001"
        assert result.source_document_title == "Test Document"
        assert isinstance(result.timestamp, datetime)


class TestPlagiarismDetector:
    """Test PlagiarismDetector class"""
    
    def setup_method(self):
        # Use a smaller model for testing
        with patch('plagiarism_detector.detector.SentenceTransformer') as mock_model:
            mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
            self.detector = PlagiarismDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        assert self.detector.model_name == "all-MiniLM-L6-v2"
        assert self.detector.thresholds['high_confidence'] == 0.8
        assert self.detector.thresholds['medium_confidence'] == 0.6
        assert self.detector.thresholds['low_confidence'] == 0.4
    
    def test_determine_verdict(self):
        """Test verdict determination"""
        # High confidence
        conf, verdict = self.detector._determine_verdict(0.85)
        assert conf == 0.9
        assert "High Probability" in verdict
        
        # Medium confidence
        conf, verdict = self.detector._determine_verdict(0.7)
        assert conf == 0.7
        assert "Possible Plagiarism" in verdict
        
        # Low confidence
        conf, verdict = self.detector._determine_verdict(0.5)
        assert conf == 0.5
        assert "Low Similarity" in verdict
        
        # Original
        conf, verdict = self.detector._determine_verdict(0.3)
        assert conf == 0.1
        assert "Likely Original" in verdict
    
    @patch('plagiarism_detector.detector.cosine_similarity')
    def test_detect_plagiarism_tfidf(self, mock_cosine):
        """Test TF-IDF plagiarism detection"""
        mock_cosine.return_value = [[0.8, 0.3, 0.1]]
        
        submission = "Test submission"
        references = ["Reference 1", "Reference 2", "Reference 3"]
        metadata = [
            DocumentMetadata(id="doc1", title="Doc 1", content="Reference 1"),
            DocumentMetadata(id="doc2", title="Doc 2", content="Reference 2"),
            DocumentMetadata(id="doc3", title="Doc 3", content="Reference 3")
        ]
        
        results = self.detector.detect_plagiarism_tfidf(submission, references, metadata)
        
        assert len(results) == 3
        assert results[0].similarity_score == 0.8
        assert results[0].detection_method == "tfidf"
        assert results[0].source_document_id == "doc1"
    
    def test_batch_detect(self):
        """Test batch detection"""
        submissions = ["Text 1", "Text 2"]
        references = ["Reference 1", "Reference 2"]
        
        with patch.object(self.detector, 'detect_plagiarism_hybrid') as mock_detect:
            mock_detect.return_value = [
                PlagiarismResult(0.5, 0.6, "Test", [], "doc1", "Doc 1")
            ]
            
            results = self.detector.batch_detect(submissions, references, method="hybrid")
            
            assert len(results) == 2
            assert "0" in results
            assert "1" in results


class TestDocumentDatabase:
    """Test DocumentDatabase class"""
    
    def setup_method(self):
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = DocumentDatabase(self.temp_db.name)
    
    def teardown_method(self):
        # Clean up temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert os.path.exists(self.temp_db.name)
    
    def test_add_and_get_document(self):
        """Test adding and retrieving documents"""
        doc = DocumentMetadata(
            id="test_001",
            title="Test Document",
            content="This is test content",
            author="Test Author"
        )
        
        doc_id = self.db.add_document(doc)
        assert doc_id == "test_001"
        
        retrieved_doc = self.db.get_document("test_001")
        assert retrieved_doc is not None
        assert retrieved_doc.title == "Test Document"
        assert retrieved_doc.content == "This is test content"
        assert retrieved_doc.author == "Test Author"
    
    def test_get_all_documents(self):
        """Test retrieving all documents"""
        # Add multiple documents
        for i in range(3):
            doc = DocumentMetadata(
                id=f"test_{i:03d}",
                title=f"Test Document {i}",
                content=f"Content {i}"
            )
            self.db.add_document(doc)
        
        documents = self.db.get_all_documents()
        assert len(documents) == 3
    
    def test_search_documents(self):
        """Test document search"""
        doc1 = DocumentMetadata(
            id="test_001",
            title="Machine Learning Basics",
            content="This is about machine learning algorithms"
        )
        doc2 = DocumentMetadata(
            id="test_002", 
            title="Data Science",
            content="This is about data analysis and statistics"
        )
        
        self.db.add_document(doc1)
        self.db.add_document(doc2)
        
        # Search by title
        results = self.db.search_documents("machine learning")
        assert len(results) == 1
        assert results[0].id == "test_001"
        
        # Search by content
        results = self.db.search_documents("statistics")
        assert len(results) == 1
        assert results[0].id == "test_002"
    
    def test_delete_document(self):
        """Test document deletion"""
        doc = DocumentMetadata(
            id="test_001",
            title="Test Document",
            content="Test content"
        )
        
        self.db.add_document(doc)
        assert self.db.get_document("test_001") is not None
        
        success = self.db.delete_document("test_001")
        assert success is True
        assert self.db.get_document("test_001") is None
        
        # Try to delete non-existent document
        success = self.db.delete_document("non_existent")
        assert success is False
    
    def test_add_submission(self):
        """Test adding submissions"""
        submission_id = self.db.add_submission("Test submission content", "Test Author")
        
        assert submission_id is not None
        assert len(submission_id) > 0
    
    def test_save_detection_results(self):
        """Test saving detection results"""
        submission_id = self.db.add_submission("Test content")
        
        results = [
            PlagiarismResult(
                similarity_score=0.8,
                confidence=0.9,
                verdict="High plagiarism",
                matched_sentences=["Sentence 1"],
                source_document_id="doc_001"
            )
        ]
        
        self.db.save_detection_results(submission_id, results)
        
        history = self.db.get_detection_history(submission_id)
        assert len(history) == 1
        assert history[0]['similarity_score'] == 0.8
        assert history[0]['verdict'] == "High plagiarism"
    
    def test_get_statistics(self):
        """Test getting statistics"""
        # Add some test data
        doc = DocumentMetadata(id="doc1", title="Test", content="Test content")
        self.db.add_document(doc)
        
        submission_id = self.db.add_submission("Test submission")
        
        result = PlagiarismResult(0.7, 0.8, "Test", [], "doc1")
        self.db.save_detection_results(submission_id, [result])
        
        stats = self.db.get_statistics()
        
        assert stats['total_documents'] == 1
        assert stats['total_submissions'] == 1
        assert stats['total_detections'] == 1
        assert stats['average_similarity'] == 0.7
        assert stats['plagiarism_count'] == 1  # Above 0.6 threshold


class TestSampleData:
    """Test sample data creation"""
    
    def test_create_sample_documents(self):
        """Test sample document creation"""
        docs = create_sample_documents()
        
        assert len(docs) == 4
        assert all(isinstance(doc, DocumentMetadata) for doc in docs)
        assert all(doc.id.startswith("doc_") for doc in docs)
        assert all(doc.word_count > 0 for doc in docs)


class TestIntegration:
    """Integration tests"""
    
    def setup_method(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = DocumentDatabase(self.temp_db.name)
        
        # Add sample documents
        self.db.populate_sample_data()
    
    def teardown_method(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    @patch('plagiarism_detector.detector.SentenceTransformer')
    @patch('plagiarism_detector.detector.cosine_similarity')
    def test_end_to_end_plagiarism_detection(self, mock_cosine, mock_model):
        """Test end-to-end plagiarism detection"""
        # Mock the sentence transformer
        mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_cosine.return_value = [[0.85, 0.3, 0.1]]
        
        detector = PlagiarismDetector()
        
        # Get reference documents
        reference_docs = self.db.get_all_documents()
        reference_documents = [doc.content for doc in reference_docs]
        
        # Test submission
        submission = "Machine learning uses statistical methods to enable computers to learn from data."
        
        # Perform detection
        results = detector.detect_plagiarism_hybrid(
            submission, reference_documents, reference_docs
        )
        
        # Save results
        submission_id = self.db.add_submission(submission)
        self.db.save_detection_results(submission_id, results)
        
        # Verify results
        assert len(results) == len(reference_docs)
        assert submission_id is not None
        
        # Check database
        history = self.db.get_detection_history(submission_id)
        assert len(history) == len(reference_docs)
        
        stats = self.db.get_statistics()
        assert stats['total_submissions'] == 1
        assert stats['total_detections'] == len(reference_docs)


# Performance tests
class TestPerformance:
    """Performance tests"""
    
    def test_large_document_processing(self):
        """Test processing large documents"""
        # Create a large document
        large_content = "This is a test sentence. " * 1000
        
        doc = DocumentMetadata(
            id="large_doc",
            title="Large Document",
            content=large_content
        )
        
        # Should not raise an exception
        assert doc.word_count > 1000
        assert doc.readability_score > 0
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        detector = PlagiarismDetector()
        
        # Create multiple submissions
        submissions = [f"Test submission {i}" for i in range(10)]
        references = ["Reference document"]
        
        with patch.object(detector, 'detect_plagiarism_hybrid') as mock_detect:
            mock_detect.return_value = [
                PlagiarismResult(0.5, 0.6, "Test", [], "doc1", "Doc 1")
            ]
            
            import time
            start_time = time.time()
            
            results = detector.batch_detect(submissions, references)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete in reasonable time (less than 1 second for mocked operations)
            assert processing_time < 1.0
            assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
