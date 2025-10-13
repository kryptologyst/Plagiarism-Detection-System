"""
FastAPI web interface for plagiarism detection system
Provides REST API endpoints and web UI for document management and plagiarism detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path

from .detector import PlagiarismDetector, DocumentMetadata, PlagiarismResult
from .database import DocumentDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plagiarism Detection System",
    description="A modern plagiarism detection system using advanced NLP techniques",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
detector = PlagiarismDetector()
database = DocumentDatabase()

# Pydantic models for API
class DocumentCreate(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source")

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    author: Optional[str]
    source: Optional[str]
    word_count: int
    readability_score: float
    created_at: datetime

class PlagiarismCheckRequest(BaseModel):
    content: str = Field(..., description="Text to check for plagiarism")
    method: str = Field("hybrid", description="Detection method: sentence_transformer, tfidf, or hybrid")
    threshold: Optional[float] = Field(None, description="Custom similarity threshold")

class PlagiarismCheckResponse(BaseModel):
    submission_id: str
    results: List[Dict[str, Any]]
    overall_verdict: str
    max_similarity: float
    detection_method: str
    timestamp: datetime

class BatchCheckRequest(BaseModel):
    submissions: List[str] = Field(..., description="List of texts to check")
    method: str = Field("hybrid", description="Detection method")
    threshold: Optional[float] = Field(None, description="Custom similarity threshold")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return get_web_interface()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get all documents"""
    try:
        documents = database.get_all_documents()
        return [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                author=doc.author,
                source=doc.source,
                word_count=doc.word_count,
                readability_score=doc.readability_score,
                created_at=doc.created_at
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.post("/api/documents", response_model=DocumentResponse)
async def create_document(document: DocumentCreate):
    """Create a new document"""
    try:
        doc_metadata = DocumentMetadata(
            id=str(uuid.uuid4()),
            title=document.title,
            content=document.content,
            author=document.author,
            source=document.source
        )
        
        doc_id = database.add_document(doc_metadata)
        created_doc = database.get_document(doc_id)
        
        return DocumentResponse(
            id=created_doc.id,
            title=created_doc.title,
            content=created_doc.content,
            author=created_doc.author,
            source=created_doc.source,
            word_count=created_doc.word_count,
            readability_score=created_doc.readability_score,
            created_at=created_doc.created_at
        )
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document")

@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document by ID"""
    try:
        document = database.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            id=document.id,
            title=document.title,
            content=document.content,
            author=document.author,
            source=document.source,
            word_count=document.word_count,
            readability_score=document.readability_score,
            created_at=document.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = database.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/api/documents/search")
async def search_documents(query: str = Form(...), limit: int = Form(10)):
    """Search documents by title or content"""
    try:
        documents = database.search_documents(query, limit)
        return [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                author=doc.author,
                source=doc.source,
                word_count=doc.word_count,
                readability_score=doc.readability_score,
                created_at=doc.created_at
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to search documents")

@app.post("/api/plagiarism/check", response_model=PlagiarismCheckResponse)
async def check_plagiarism(request: PlagiarismCheckRequest):
    """Check a single text for plagiarism"""
    try:
        # Get all reference documents
        reference_docs = database.get_all_documents()
        if not reference_docs:
            raise HTTPException(status_code=400, detail="No reference documents available")
        
        reference_documents = [doc.content for doc in reference_docs]
        
        # Perform detection
        if request.method == "sentence_transformer":
            results = detector.detect_plagiarism_sentence_transformer(
                request.content, reference_documents, reference_docs
            )
        elif request.method == "tfidf":
            results = detector.detect_plagiarism_tfidf(
                request.content, reference_documents, reference_docs
            )
        else:  # hybrid
            results = detector.detect_plagiarism_hybrid(
                request.content, reference_documents, reference_docs
            )
        
        # Save submission and results
        submission_id = database.add_submission(request.content)
        database.save_detection_results(submission_id, results)
        
        # Prepare response
        result_dicts = []
        max_similarity = 0.0
        overall_verdict = "✅ Likely Original"
        
        for result in results:
            if result.similarity_score > max_similarity:
                max_similarity = result.similarity_score
                overall_verdict = result.verdict
            
            result_dicts.append({
                "similarity_score": result.similarity_score,
                "confidence": result.confidence,
                "verdict": result.verdict,
                "matched_sentences": result.matched_sentences,
                "source_document_id": result.source_document_id,
                "source_document_title": result.source_document_title,
                "detection_method": result.detection_method
            })
        
        return PlagiarismCheckResponse(
            submission_id=submission_id,
            results=result_dicts,
            overall_verdict=overall_verdict,
            max_similarity=max_similarity,
            detection_method=request.method,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking plagiarism: {e}")
        raise HTTPException(status_code=500, detail="Failed to check plagiarism")

@app.post("/api/plagiarism/batch-check")
async def batch_check_plagiarism(request: BatchCheckRequest):
    """Check multiple texts for plagiarism"""
    try:
        # Get all reference documents
        reference_docs = database.get_all_documents()
        if not reference_docs:
            raise HTTPException(status_code=400, detail="No reference documents available")
        
        reference_documents = [doc.content for doc in reference_docs]
        
        # Perform batch detection
        batch_results = detector.batch_detect(
            request.submissions, reference_documents, reference_docs, request.method
        )
        
        # Save submissions and results
        submission_ids = []
        for idx, submission in enumerate(request.submissions):
            submission_id = database.add_submission(submission)
            submission_ids.append(submission_id)
            database.save_detection_results(submission_id, batch_results[str(idx)])
        
        return {
            "submission_ids": submission_ids,
            "results": batch_results,
            "detection_method": request.method,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch plagiarism check: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform batch check")

@app.get("/api/plagiarism/history")
async def get_detection_history(submission_id: Optional[str] = None, limit: int = 50):
    """Get plagiarism detection history"""
    try:
        history = database.get_detection_history(submission_id, limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve detection history")

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = database.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(None),
    source: str = Form(None)
):
    """Upload a document file"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        doc_metadata = DocumentMetadata(
            id=str(uuid.uuid4()),
            title=title,
            content=content_str,
            author=author,
            source=source or file.filename
        )
        
        doc_id = database.add_document(doc_metadata)
        return {"message": "Document uploaded successfully", "document_id": doc_id}
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@app.post("/api/export")
async def export_data():
    """Export all data"""
    try:
        export_path = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        database.export_data(export_path)
        
        return {"message": "Data exported successfully", "file": export_path}
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

def get_web_interface() -> str:
    """Generate HTML interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .tab:hover {
            background: #f8f9ff;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            min-height: 150px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .result-item {
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .similarity-score {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .high-risk { color: #dc3545; }
        .medium-risk { color: #ffc107; }
        .low-risk { color: #28a745; }
        
        .document-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .document-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .document-card h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .document-card p {
            color: #666;
            margin-bottom: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Plagiarism Detection System</h1>
            <p>Advanced NLP-powered plagiarism detection using state-of-the-art techniques</p>
        </div>
        
        <div class="content">
            <div class="tabs">
                <div class="tab active" onclick="showTab('check')">Check Plagiarism</div>
                <div class="tab" onclick="showTab('documents')">Manage Documents</div>
                <div class="tab" onclick="showTab('history')">Detection History</div>
                <div class="tab" onclick="showTab('statistics')">Statistics</div>
            </div>
            
            <!-- Check Plagiarism Tab -->
            <div id="check" class="tab-content active">
                <h2>Check for Plagiarism</h2>
                <form id="plagiarismForm">
                    <div class="form-group">
                        <label for="content">Text to Check:</label>
                        <textarea id="content" placeholder="Enter the text you want to check for plagiarism..." required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="method">Detection Method:</label>
                        <select id="method">
                            <option value="hybrid">Hybrid (Recommended)</option>
                            <option value="sentence_transformer">Sentence Transformer</option>
                            <option value="tfidf">TF-IDF</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">Check for Plagiarism</button>
                </form>
                
                <div id="checkResults"></div>
            </div>
            
            <!-- Manage Documents Tab -->
            <div id="documents" class="tab-content">
                <h2>Document Management</h2>
                
                <div style="display: flex; gap: 20px; margin-bottom: 30px;">
                    <button class="btn" onclick="showDocumentForm()">Add New Document</button>
                    <button class="btn btn-secondary" onclick="loadDocuments()">Refresh Documents</button>
                </div>
                
                <div id="documentForm" style="display: none; background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                    <h3>Add New Document</h3>
                    <form id="addDocumentForm">
                        <div class="form-group">
                            <label for="docTitle">Title:</label>
                            <input type="text" id="docTitle" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="docContent">Content:</label>
                            <textarea id="docContent" required></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="docAuthor">Author:</label>
                            <input type="text" id="docAuthor">
                        </div>
                        
                        <div class="form-group">
                            <label for="docSource">Source:</label>
                            <input type="text" id="docSource">
                        </div>
                        
                        <button type="submit" class="btn">Add Document</button>
                        <button type="button" class="btn btn-secondary" onclick="hideDocumentForm()">Cancel</button>
                    </form>
                </div>
                
                <div id="documentsList"></div>
            </div>
            
            <!-- Detection History Tab -->
            <div id="history" class="tab-content">
                <h2>Detection History</h2>
                <div id="historyContent"></div>
            </div>
            
            <!-- Statistics Tab -->
            <div id="statistics" class="tab-content">
                <h2>System Statistics</h2>
                <div id="statisticsContent"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Load content based on tab
            if (tabName === 'documents') {
                loadDocuments();
            } else if (tabName === 'history') {
                loadHistory();
            } else if (tabName === 'statistics') {
                loadStatistics();
            }
        }
        
        // Plagiarism check form
        document.getElementById('plagiarismForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const content = document.getElementById('content').value;
            const method = document.getElementById('method').value;
            
            const resultsDiv = document.getElementById('checkResults');
            resultsDiv.innerHTML = '<div class="loading">Checking for plagiarism...</div>';
            
            try {
                const response = await fetch('/api/plagiarism/check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        method: method
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayPlagiarismResults(data);
                } else {
                    resultsDiv.innerHTML = `<div class="error">Error: ${data.detail}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        });
        
        function displayPlagiarismResults(data) {
            const resultsDiv = document.getElementById('checkResults');
            
            let html = `
                <div class="results">
                    <h3>Detection Results</h3>
                    <p><strong>Overall Verdict:</strong> ${data.overall_verdict}</p>
                    <p><strong>Max Similarity:</strong> ${(data.max_similarity * 100).toFixed(1)}%</p>
                    <p><strong>Method:</strong> ${data.detection_method}</p>
                    <p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                </div>
            `;
            
            data.results.forEach((result, index) => {
                const riskClass = result.similarity_score > 0.8 ? 'high-risk' : 
                                result.similarity_score > 0.6 ? 'medium-risk' : 'low-risk';
                
                html += `
                    <div class="result-item">
                        <div class="similarity-score ${riskClass}">
                            Similarity: ${(result.similarity_score * 100).toFixed(1)}%
                        </div>
                        <p><strong>Verdict:</strong> ${result.verdict}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Source:</strong> ${result.source_document_title || 'Unknown'}</p>
                        ${result.matched_sentences.length > 0 ? 
                            `<p><strong>Matched Sentences:</strong> ${result.matched_sentences.length}</p>` : 
                            ''
                        }
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        // Document management
        function showDocumentForm() {
            document.getElementById('documentForm').style.display = 'block';
        }
        
        function hideDocumentForm() {
            document.getElementById('documentForm').style.display = 'none';
            document.getElementById('addDocumentForm').reset();
        }
        
        document.getElementById('addDocumentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const title = document.getElementById('docTitle').value;
            const content = document.getElementById('docContent').value;
            const author = document.getElementById('docAuthor').value;
            const source = document.getElementById('docSource').value;
            
            try {
                const response = await fetch('/api/documents', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        title: title,
                        content: content,
                        author: author,
                        source: source
                    })
                });
                
                if (response.ok) {
                    hideDocumentForm();
                    loadDocuments();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
        
        async function loadDocuments() {
            const documentsDiv = document.getElementById('documentsList');
            documentsDiv.innerHTML = '<div class="loading">Loading documents...</div>';
            
            try {
                const response = await fetch('/api/documents');
                const documents = await response.json();
                
                if (documents.length === 0) {
                    documentsDiv.innerHTML = '<p>No documents found. Add some documents to get started!</p>';
                    return;
                }
                
                let html = '<div class="document-list">';
                documents.forEach(doc => {
                    html += `
                        <div class="document-card">
                            <h3>${doc.title}</h3>
                            <p><strong>Author:</strong> ${doc.author || 'Unknown'}</p>
                            <p><strong>Source:</strong> ${doc.source || 'Unknown'}</p>
                            <p><strong>Word Count:</strong> ${doc.word_count}</p>
                            <p><strong>Readability:</strong> ${doc.readability_score.toFixed(1)}</p>
                            <p><strong>Created:</strong> ${new Date(doc.created_at).toLocaleDateString()}</p>
                            <button class="btn btn-secondary" onclick="deleteDocument('${doc.id}')" style="margin-top: 10px;">Delete</button>
                        </div>
                    `;
                });
                html += '</div>';
                
                documentsDiv.innerHTML = html;
            } catch (error) {
                documentsDiv.innerHTML = `<div class="error">Error loading documents: ${error.message}</div>`;
            }
        }
        
        async function deleteDocument(docId) {
            if (!confirm('Are you sure you want to delete this document?')) {
                return;
            }
            
            try {
                const response = await fetch(`/api/documents/${docId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    loadDocuments();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        // Load history
        async function loadHistory() {
            const historyDiv = document.getElementById('historyContent');
            historyDiv.innerHTML = '<div class="loading">Loading detection history...</div>';
            
            try {
                const response = await fetch('/api/plagiarism/history');
                const data = await response.json();
                
                if (data.history.length === 0) {
                    historyDiv.innerHTML = '<p>No detection history found.</p>';
                    return;
                }
                
                let html = '<div class="results">';
                data.history.forEach(item => {
                    const riskClass = item.similarity_score > 0.8 ? 'high-risk' : 
                                    item.similarity_score > 0.6 ? 'medium-risk' : 'low-risk';
                    
                    html += `
                        <div class="result-item">
                            <div class="similarity-score ${riskClass}">
                                Similarity: ${(item.similarity_score * 100).toFixed(1)}%
                            </div>
                            <p><strong>Verdict:</strong> ${item.verdict}</p>
                            <p><strong>Method:</strong> ${item.detection_method}</p>
                            <p><strong>Reference:</strong> ${item.reference_title || 'Unknown'}</p>
                            <p><strong>Date:</strong> ${new Date(item.created_at).toLocaleString()}</p>
                        </div>
                    `;
                });
                html += '</div>';
                
                historyDiv.innerHTML = html;
            } catch (error) {
                historyDiv.innerHTML = `<div class="error">Error loading history: ${error.message}</div>`;
            }
        }
        
        // Load statistics
        async function loadStatistics() {
            const statsDiv = document.getElementById('statisticsContent');
            statsDiv.innerHTML = '<div class="loading">Loading statistics...</div>';
            
            try {
                const response = await fetch('/api/statistics');
                const stats = await response.json();
                
                let html = `
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">${stats.total_documents}</div>
                            <div class="stat-label">Total Documents</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.total_submissions}</div>
                            <div class="stat-label">Total Submissions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.total_detections}</div>
                            <div class="stat-label">Total Detections</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${(stats.average_similarity * 100).toFixed(1)}%</div>
                            <div class="stat-label">Average Similarity</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.plagiarism_rate}%</div>
                            <div class="stat-label">Plagiarism Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.plagiarism_count}</div>
                            <div class="stat-label">Plagiarism Cases</div>
                        </div>
                    </div>
                `;
                
                statsDiv.innerHTML = html;
            } catch (error) {
                statsDiv.innerHTML = `<div class="error">Error loading statistics: ${error.message}</div>`;
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadDocuments();
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
