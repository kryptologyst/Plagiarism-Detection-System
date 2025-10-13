"""
Database layer for plagiarism detection system
Handles document storage, retrieval, and metadata management using SQLite
"""

import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid

from .detector import DocumentMetadata, PlagiarismResult

logger = logging.getLogger(__name__)


class DocumentDatabase:
    """SQLite database for storing documents and plagiarism detection results"""
    
    def __init__(self, db_path: str = "plagiarism_detector.db"):
        """
        Initialize the database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    source TEXT,
                    word_count INTEGER,
                    readability_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detection results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_results (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT,
                    reference_document_id TEXT,
                    similarity_score REAL,
                    confidence REAL,
                    verdict TEXT,
                    detection_method TEXT,
                    matched_sentences TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (reference_document_id) REFERENCES documents (id)
                )
            """)
            
            # Submissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    author TEXT,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_submission ON detection_results(submission_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_reference ON detection_results(reference_document_id)")
            
            conn.commit()
    
    def add_document(self, document: DocumentMetadata) -> str:
        """
        Add a document to the database
        
        Args:
            document: DocumentMetadata object
            
        Returns:
            Document ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, content, author, source, word_count, readability_score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document.id,
                document.title,
                document.content,
                document.author,
                document.source,
                document.word_count,
                document.readability_score,
                document.created_at,
                datetime.now()
            ))
            
            conn.commit()
            return document.id
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get a document by ID
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentMetadata object or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, author, source, word_count, readability_score, created_at
                FROM documents WHERE id = ?
            """, (document_id,))
            
            row = cursor.fetchone()
            if row:
                return DocumentMetadata(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    author=row[3],
                    source=row[4],
                    word_count=row[5],
                    readability_score=row[6],
                    created_at=datetime.fromisoformat(row[7])
                )
            return None
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        """
        Get all documents from the database
        
        Returns:
            List of DocumentMetadata objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, author, source, word_count, readability_score, created_at
                FROM documents ORDER BY created_at DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append(DocumentMetadata(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    author=row[3],
                    source=row[4],
                    word_count=row[5],
                    readability_score=row[6],
                    created_at=datetime.fromisoformat(row[7])
                ))
            
            return documents
    
    def search_documents(self, query: str, limit: int = 10) -> List[DocumentMetadata]:
        """
        Search documents by title or content
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching DocumentMetadata objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, author, source, word_count, readability_score, created_at
                FROM documents 
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            documents = []
            for row in cursor.fetchall():
                documents.append(DocumentMetadata(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    author=row[3],
                    source=row[4],
                    word_count=row[5],
                    readability_score=row[6],
                    created_at=datetime.fromisoformat(row[7])
                ))
            
            return documents
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the database
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def add_submission(self, content: str, author: Optional[str] = None, title: Optional[str] = None) -> str:
        """
        Add a submission to the database
        
        Args:
            content: Submission content
            author: Author name
            title: Submission title
            
        Returns:
            Submission ID
        """
        submission_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO submissions (id, content, author, title, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (submission_id, content, author, title, datetime.now()))
            
            conn.commit()
            return submission_id
    
    def save_detection_results(self, submission_id: str, results: List[PlagiarismResult]):
        """
        Save plagiarism detection results to the database
        
        Args:
            submission_id: ID of the submission
            results: List of PlagiarismResult objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                result_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT INTO detection_results 
                    (id, submission_id, reference_document_id, similarity_score, confidence, 
                     verdict, detection_method, matched_sentences, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id,
                    submission_id,
                    result.source_document_id,
                    result.similarity_score,
                    result.confidence,
                    result.verdict,
                    result.detection_method,
                    json.dumps(result.matched_sentences),
                    result.timestamp
                ))
            
            conn.commit()
    
    def get_detection_history(self, submission_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get detection history
        
        Args:
            submission_id: Optional submission ID to filter by
            limit: Maximum number of results
            
        Returns:
            List of detection result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if submission_id:
                cursor.execute("""
                    SELECT dr.*, d.title as reference_title, s.title as submission_title
                    FROM detection_results dr
                    LEFT JOIN documents d ON dr.reference_document_id = d.id
                    LEFT JOIN submissions s ON dr.submission_id = s.id
                    WHERE dr.submission_id = ?
                    ORDER BY dr.created_at DESC
                    LIMIT ?
                """, (submission_id, limit))
            else:
                cursor.execute("""
                    SELECT dr.*, d.title as reference_title, s.title as submission_title
                    FROM detection_results dr
                    LEFT JOIN documents d ON dr.reference_document_id = d.id
                    LEFT JOIN submissions s ON dr.submission_id = s.id
                    ORDER BY dr.created_at DESC
                    LIMIT ?
                """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'submission_id': row[1],
                    'reference_document_id': row[2],
                    'similarity_score': row[3],
                    'confidence': row[4],
                    'verdict': row[5],
                    'detection_method': row[6],
                    'matched_sentences': json.loads(row[7]) if row[7] else [],
                    'created_at': row[8],
                    'reference_title': row[9],
                    'submission_title': row[10]
                })
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Count submissions
            cursor.execute("SELECT COUNT(*) FROM submissions")
            submission_count = cursor.fetchone()[0]
            
            # Count detection results
            cursor.execute("SELECT COUNT(*) FROM detection_results")
            detection_count = cursor.fetchone()[0]
            
            # Average similarity score
            cursor.execute("SELECT AVG(similarity_score) FROM detection_results")
            avg_similarity = cursor.fetchone()[0] or 0
            
            # Plagiarism rate (results above medium threshold)
            cursor.execute("""
                SELECT COUNT(*) FROM detection_results 
                WHERE similarity_score >= 0.6
            """)
            plagiarism_count = cursor.fetchone()[0]
            
            plagiarism_rate = (plagiarism_count / detection_count * 100) if detection_count > 0 else 0
            
            return {
                'total_documents': doc_count,
                'total_submissions': submission_count,
                'total_detections': detection_count,
                'average_similarity': round(avg_similarity, 3),
                'plagiarism_rate': round(plagiarism_rate, 2),
                'plagiarism_count': plagiarism_count
            }
    
    def populate_sample_data(self):
        """Populate database with sample documents"""
        from .detector import create_sample_documents
        
        sample_docs = create_sample_documents()
        
        for doc in sample_docs:
            self.add_document(doc)
        
        logger.info(f"Added {len(sample_docs)} sample documents to database")
    
    def export_data(self, export_path: str):
        """
        Export database data to JSON file
        
        Args:
            export_path: Path to export file
        """
        data = {
            'documents': [],
            'submissions': [],
            'detection_results': [],
            'statistics': self.get_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Export documents
        documents = self.get_all_documents()
        for doc in documents:
            data['documents'].append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'author': doc.author,
                'source': doc.source,
                'word_count': doc.word_count,
                'readability_score': doc.readability_score,
                'created_at': doc.created_at.isoformat()
            })
        
        # Export submissions
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM submissions")
            for row in cursor.fetchall():
                data['submissions'].append({
                    'id': row[0],
                    'content': row[1],
                    'author': row[2],
                    'title': row[3],
                    'created_at': row[4]
                })
        
        # Export detection results
        detection_history = self.get_detection_history(limit=1000)
        data['detection_results'] = detection_history
        
        # Write to file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Database exported to {export_path}")


if __name__ == "__main__":
    # Example usage
    db = DocumentDatabase("test_plagiarism.db")
    
    # Populate with sample data
    db.populate_sample_data()
    
    # Get statistics
    stats = db.get_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Search documents
    results = db.search_documents("machine learning")
    print(f"\nFound {len(results)} documents matching 'machine learning'")
    
    for doc in results:
        print(f"  - {doc.title} by {doc.author}")
