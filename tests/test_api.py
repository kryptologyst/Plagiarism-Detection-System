"""
API tests for FastAPI endpoints
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from plagiarism_detector.api import app
from plagiarism_detector.detector import DocumentMetadata, PlagiarismResult
from plagiarism_detector.database import DocumentDatabase


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    # Patch the global database instance
    with patch('plagiarism_detector.api.database', DocumentDatabase(temp_db.name)):
        yield temp_db.name
    
    # Cleanup
    if os.path.exists(temp_db.name):
        os.unlink(temp_db.name)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    def test_get_documents_empty(self, client, temp_db):
        """Test getting documents when none exist"""
        response = client.get("/api/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert data == []
    
    def test_create_document(self, client, temp_db):
        """Test creating a document"""
        document_data = {
            "title": "Test Document",
            "content": "This is test content",
            "author": "Test Author",
            "source": "Test Source"
        }
        
        response = client.post("/api/documents", json=document_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["title"] == "Test Document"
        assert data["content"] == "This is test content"
        assert data["author"] == "Test Author"
        assert data["source"] == "Test Source"
        assert "id" in data
        assert "word_count" in data
        assert "readability_score" in data
        assert "created_at" in data
    
    def test_get_document_by_id(self, client, temp_db):
        """Test getting a specific document by ID"""
        # First create a document
        document_data = {
            "title": "Test Document",
            "content": "This is test content"
        }
        
        create_response = client.post("/api/documents", json=document_data)
        doc_id = create_response.json()["id"]
        
        # Then get it by ID
        response = client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == doc_id
        assert data["title"] == "Test Document"
    
    def test_get_nonexistent_document(self, client, temp_db):
        """Test getting a non-existent document"""
        response = client.get("/api/documents/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_delete_document(self, client, temp_db):
        """Test deleting a document"""
        # First create a document
        document_data = {
            "title": "Test Document",
            "content": "This is test content"
        }
        
        create_response = client.post("/api/documents", json=document_data)
        doc_id = create_response.json()["id"]
        
        # Then delete it
        response = client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "deleted successfully" in data["message"]
        
        # Verify it's deleted
        get_response = client.get(f"/api/documents/{doc_id}")
        assert get_response.status_code == 404
    
    def test_search_documents(self, client, temp_db):
        """Test searching documents"""
        # Create test documents
        documents = [
            {"title": "Machine Learning Guide", "content": "This is about ML algorithms"},
            {"title": "Data Science Basics", "content": "This covers data analysis"}
        ]
        
        for doc_data in documents:
            client.post("/api/documents", json=doc_data)
        
        # Search for machine learning
        response = client.post("/api/documents/search", data={"query": "machine learning", "limit": 10})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert "Machine Learning" in data[0]["title"]


class TestPlagiarismEndpoints:
    """Test plagiarism detection endpoints"""
    
    @patch('plagiarism_detector.api.detector')
    def test_check_plagiarism(self, mock_detector, client, temp_db):
        """Test plagiarism checking endpoint"""
        # Mock the detector
        mock_result = PlagiarismResult(
            similarity_score=0.8,
            confidence=0.9,
            verdict="High plagiarism",
            matched_sentences=["Sentence 1"],
            source_document_id="doc_001",
            source_document_title="Test Document"
        )
        mock_detector.detect_plagiarism_hybrid.return_value = [mock_result]
        
        # Add a reference document
        doc_data = {
            "title": "Reference Document",
            "content": "This is reference content"
        }
        client.post("/api/documents", json=doc_data)
        
        # Check plagiarism
        request_data = {
            "content": "This is test content to check",
            "method": "hybrid"
        }
        
        response = client.post("/api/plagiarism/check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "submission_id" in data
        assert "results" in data
        assert "overall_verdict" in data
        assert "max_similarity" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity_score"] == 0.8
    
    def test_check_plagiarism_no_references(self, client, temp_db):
        """Test plagiarism checking with no reference documents"""
        request_data = {
            "content": "This is test content",
            "method": "hybrid"
        }
        
        response = client.post("/api/plagiarism/check", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "no reference documents" in data["detail"].lower()
    
    @patch('plagiarism_detector.api.detector')
    def test_batch_check_plagiarism(self, mock_detector, client, temp_db):
        """Test batch plagiarism checking"""
        # Mock the detector
        mock_result = PlagiarismResult(
            similarity_score=0.7,
            confidence=0.8,
            verdict="Possible plagiarism",
            matched_sentences=[],
            source_document_id="doc_001"
        )
        mock_detector.batch_detect.return_value = {
            "0": [mock_result],
            "1": [mock_result]
        }
        
        # Add a reference document
        doc_data = {
            "title": "Reference Document",
            "content": "This is reference content"
        }
        client.post("/api/documents", json=doc_data)
        
        # Batch check
        request_data = {
            "submissions": ["Text 1", "Text 2"],
            "method": "hybrid"
        }
        
        response = client.post("/api/plagiarism/batch-check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "submission_ids" in data
        assert "results" in data
        assert len(data["submission_ids"]) == 2
        assert len(data["results"]) == 2
    
    def test_get_detection_history(self, client, temp_db):
        """Test getting detection history"""
        response = client.get("/api/plagiarism/history")
        assert response.status_code == 200
        
        data = response.json()
        assert "history" in data
        assert "count" in data
        assert isinstance(data["history"], list)
        assert isinstance(data["count"], int)


class TestStatisticsEndpoint:
    """Test statistics endpoint"""
    
    def test_get_statistics(self, client, temp_db):
        """Test getting system statistics"""
        response = client.get("/api/statistics")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            "total_documents",
            "total_submissions", 
            "total_detections",
            "average_similarity",
            "plagiarism_rate",
            "plagiarism_count"
        ]
        
        for field in required_fields:
            assert field in data


class TestWebInterface:
    """Test web interface"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Plagiarism Detection System" in response.text


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self, client, temp_db):
        """Test handling invalid JSON"""
        response = client.post(
            "/api/documents",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client, temp_db):
        """Test handling missing required fields"""
        response = client.post("/api/documents", json={"title": "Test"})
        assert response.status_code == 422
    
    def test_invalid_method(self, client, temp_db):
        """Test handling invalid detection method"""
        # Add a reference document first
        doc_data = {"title": "Test", "content": "Test content"}
        client.post("/api/documents", json=doc_data)
        
        request_data = {
            "content": "Test content",
            "method": "invalid_method"
        }
        
        response = client.post("/api/plagiarism/check", json=request_data)
        # Should still work as the method gets passed through
        assert response.status_code in [200, 400]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
