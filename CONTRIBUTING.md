# Contributing to Plagiarism Detection System

Thank you for your interest in contributing to the Plagiarism Detection System! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of Python, NLP, and web development

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/plagiarism-detector.git
   cd plagiarism-detector
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

## 🛠️ Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical fixes

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make Changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   pytest
   flake8 plagiarism_detector/
   mypy plagiarism_detector/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/amazing-feature
   ```

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for all functions and classes
- Maximum line length: 127 characters

### Code Formatting
```bash
# Format code
black plagiarism_detector/

# Sort imports
isort plagiarism_detector/

# Lint code
flake8 plagiarism_detector/
```

### Commit Message Format
Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

Examples:
```
feat: add sentence-level plagiarism detection
fix: resolve database connection issue
docs: update API documentation
```

## 🧪 Testing Guidelines

### Test Structure
- Unit tests for individual functions
- Integration tests for component interactions
- API tests for endpoints
- Performance tests for critical paths

### Writing Tests
```python
def test_plagiarism_detection():
    """Test plagiarism detection functionality"""
    detector = PlagiarismDetector()
    result = detector.detect_plagiarism_hybrid(
        submission="Test text",
        reference_documents=["Reference text"]
    )
    assert len(result) > 0
    assert result[0].similarity_score >= 0
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=plagiarism_detector

# Run specific test file
pytest tests/test_detector.py

# Run with verbose output
pytest -v
```

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Document parameters and return values
- Provide usage examples

### API Documentation
- Update OpenAPI schemas
- Document new endpoints
- Include request/response examples

### User Documentation
- Update README.md for new features
- Add usage examples
- Document configuration options

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Test with latest version
3. Gather relevant information

### Bug Report Template
```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Package Version: [e.g., 2.0.0]

**Additional Context**
Any other relevant information.
```

## ✨ Feature Requests

### Feature Request Template
```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives**
Other solutions you've considered.

**Additional Context**
Any other relevant information.
```

## 🔒 Security

### Reporting Security Issues
- **DO NOT** create public issues for security vulnerabilities
- Email security issues to: security@plagiarism-detector.com
- Include detailed information about the vulnerability
- Allow time for response before public disclosure

### Security Guidelines
- Validate all inputs
- Use parameterized queries
- Implement proper authentication
- Follow OWASP guidelines

## 🚀 Release Process

### Version Numbering
- `MAJOR.MINOR.PATCH` (Semantic Versioning)
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes prepared

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the community

### Getting Help
- Check documentation first
- Search existing issues
- Ask questions in discussions
- Join community chat (if available)

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/plagiarism-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/plagiarism-detector/discussions)
- **Email**: contributors@plagiarism-detector.com

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Plagiarism Detection System! 🎉
