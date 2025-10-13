"""
Modern Plagiarism Detection System
A comprehensive plagiarism detection tool using state-of-the-art NLP techniques
"""

from .detector import PlagiarismDetector, PlagiarismResult, DocumentMetadata
from .database import DocumentDatabase
from .api import app

__version__ = "2.0.0"
__author__ = "AI Projects"

__all__ = [
    "PlagiarismDetector",
    "PlagiarismResult", 
    "DocumentMetadata",
    "DocumentDatabase",
    "app"
]
