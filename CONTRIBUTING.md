# Contributing to CRLLM

Thank you for your interest in contributing to CRLLM! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/CRLLM.git
   cd CRLLM
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-username/CRLLM.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key (for testing)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Optional: export DWAVE_API_TOKEN="your-dwave-token"
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=crllm --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Format code
black crllm/ tests/ examples/

# Sort imports
isort crllm/ tests/ examples/

# Lint code
flake8 crllm/ tests/ examples/

# Type checking
mypy crllm/

# Security check
bandit -r crllm/
```

## Contributing Guidelines

### Types of Contributions

We welcome contributions in the following areas:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code
6. **Examples**: Add new examples or demos

### Before You Start

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** for significant changes to discuss the approach
3. **Keep changes focused** - one feature or bug fix per pull request
4. **Write tests** for new functionality
5. **Update documentation** as needed

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation
   - Ensure all tests pass

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

## Code Style

### Python Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Code Formatting

```bash
# Format all Python files
black crllm/ tests/ examples/

# Check formatting without making changes
black --check crllm/ tests/ examples/
```

### Import Organization

```bash
# Sort imports
isort crllm/ tests/ examples/

# Check import sorting
isort --check-only crllm/ tests/ examples/
```

### Linting

```bash
# Run flake8
flake8 crllm/ tests/ examples/

# Run with specific configuration
flake8 --config=.flake8 crllm/
```

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types
- Run `mypy` to check type consistency

Example:
```python
from typing import Dict, List, Optional, Union

def process_query(
    query: str,
    domain: Optional[str] = None,
    config: Dict[str, Any] = None
) -> ReasoningResult:
    """Process a query through the CRLLM pipeline."""
    pass
```

### Documentation

- Use Google-style docstrings
- Include type information in docstrings
- Document all public functions and classes
- Include examples in docstrings when helpful

Example:
```python
def process_query(
    self,
    query: str,
    domain: Optional[str] = None,
    **kwargs
) -> ReasoningResult:
    """Process a query through the CRLLM pipeline.
    
    Args:
        query: Input query string
        domain: Optional reasoning domain
        **kwargs: Additional parameters
        
    Returns:
        ReasoningResult containing the answer and metadata
        
    Example:
        >>> pipeline = CRLLMPipeline()
        >>> result = pipeline.process_query("Why does smoking cause cancer?")
        >>> print(result.final_answer)
    """
    pass
```

## Testing

### Test Structure

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use descriptive test names
- Group related tests in classes

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test module interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from crllm.core import CRLLMPipeline

class TestCRLLMPipeline:
    """Test cases for CRLLMPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default modules."""
        pipeline = CRLLMPipeline()
        assert pipeline.task_interface is not None
        assert pipeline.reason_sampler is not None
    
    @patch('crllm.core.openai.ChatCompletion.create')
    def test_process_query_with_mock(self, mock_openai):
        """Test query processing with mocked OpenAI."""
        mock_openai.return_value.choices = [Mock()]
        mock_openai.return_value.choices[0].message.content = "Test answer"
        
        pipeline = CRLLMPipeline()
        result = pipeline.process_query("Test query")
        
        assert result.final_answer == "Test answer"
```

### Test Coverage

- Aim for >90% test coverage
- Test both success and failure cases
- Test edge cases and boundary conditions
- Use fixtures for common test data

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=crllm --cov-report=html

# Run specific test
pytest tests/test_core.py::TestCRLLMPipeline::test_pipeline_initialization

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Run code quality checks**:
   ```bash
   black --check crllm/ tests/ examples/
   isort --check-only crllm/ tests/ examples/
   flake8 crllm/ tests/ examples/
   mypy crllm/
   ```

3. **Update documentation** if needed

4. **Add changelog entry** if applicable

### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review** if applicable

### After Approval

1. **Squash commits** if requested
2. **Rebase** on latest main branch
3. **Address feedback** promptly
4. **Monitor CI/CD** results

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details** (OS, Python version, etc.)
5. **Error messages** and logs
6. **Minimal code example** if applicable

### Feature Requests

For feature requests, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**
5. **Additional context**

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information is requested

## Documentation

### Documentation Types

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: Generated from docstrings
3. **User Guides**: README, tutorials, examples
4. **Developer Documentation**: Contributing guidelines, architecture

### Documentation Standards

- Use clear, concise language
- Include examples and code snippets
- Keep documentation up-to-date
- Use consistent formatting
- Include diagrams when helpful

### Building Documentation

```bash
# Build Sphinx documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `setup.py` and `__init__.py`
2. **Update changelog** with new features and fixes
3. **Run full test suite** to ensure everything works
4. **Update documentation** if needed
5. **Create release tag** on GitHub
6. **Publish to PyPI** (maintainers only)

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: support@crllm.com

### Resources

- [Python Documentation](https://docs.python.org/3/)
- [Gradio Documentation](https://gradio.app/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to CRLLM! 🚀
